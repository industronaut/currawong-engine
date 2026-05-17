//! View-side rendering for the lumber camp.
//!
//! Three [`RenderId`] templates (pawn capsule, tree cone, stockpile cube)
//! drawn through one PBR pipeline against a fixed afternoon sun. Each template
//! owns its own mesh buffers + a PBR material instance (albedo factor +
//! metallic/roughness) so the draw loop is just one indexed-instanced call
//! per template.
//!
//! Per-instance hit IDs feed [`Renderer::hit_id_hover`]: the hovered object
//! gets a warm-gold tint, and left-clicking a tree toggles a [`Designated`]
//! component on it. Every designated tree gets a small red downward-pointing
//! cone floating above its apex; every [`Carrying`](crate::sim::Carrying)
//! pawn gets a brown log riding on their shoulders (handled inside
//! [`pawn::PawnRenderer`]).
//!
//! Layout: this module owns the orchestration ([`LumberCampView`], the
//! fused render walk, shared infrastructure like [`MeshTemplate`]) and
//! per-kind state lives in sibling submodules — [`pawn`], [`tree`],
//! [`stockpile`]. Adding a new kind is a new sibling module + a registration
//! line in [`init`](LumberCampView::init) + an arm in the dispatch `match`.

mod pawn;
mod stockpile;
mod tree;

use std::collections::HashMap;
use std::time::Duration;

use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, HitTarget, InstanceBuckets,
    MeshInstanceAttribs, OrbitRig, PbrMaterial, PbrMaterialInstance, PbrMaterialParams,
    PrimitiveMesh, Renderer, SamplerKind, SamplerRegistry, TerrainMaterial,
    TerrainMaterialInstance, TerrainRenderer, Texture, View, ViewConfig, ViewEnvironment,
    WorldObjectRef, ZoneId, wgpu, winit, yakui,
};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use pawn::PawnRenderer;
use tree::TreeRenderer;

use crate::sim::{Game, GameState, HEIGHT_UNIT, RenderId, TILE_SIZE, TIME_LIMIT_SECS, WOOD_GOAL};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES_PER_TEMPLATE: u32 = 64;
const TERRAIN_TINT: Vec4 = Vec4::new(0.55, 0.65, 0.42, 1.0); // grass green
/// Per-instance multiplier applied to the hovered object's albedo. Boosted
/// above one so the highlight overdrives the lit colour, warm-gold biased
/// so the signal reads as "selectable" against the neutral scene palette.
pub(crate) const HOVER_TINT: Vec4 = Vec4::new(1.8, 1.6, 0.8, 1.0);

// --- Per-template GPU resources ----------------------------------------
//
// Shared infrastructure: every per-kind submodule builds at least one
// `MeshTemplate`, and most also push into the shared `InstanceBuckets`
// keyed by `RenderId`. Kept `pub(super)`-flavoured (just `pub` here since
// the binary crate has no external API) so the submodules can construct
// templates without re-exporting builders.

/// Mesh buffers + the PBR material instance to draw them with. One per
/// [`RenderId`] kind; bound together because the draw loop always swaps
/// both at once.
pub struct MeshTemplate {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,
    pub material: PbrMaterialInstance,
}

impl MeshTemplate {
    pub fn new(
        renderer: &Renderer,
        material: &PbrMaterial,
        samplers: &SamplerRegistry,
        albedo: &Texture,
        mesh: &PrimitiveMesh,
        params: TemplateParams,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let vertices = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(params.label),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let indices = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(params.label),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let material = material.create_instance(
            renderer,
            samplers,
            PbrMaterialParams {
                albedo,
                sampler: SamplerKind::LinearClamp,
                albedo_factor: params.albedo_factor,
                metallic: params.metallic,
                roughness: params.roughness,
            },
        );
        Self {
            vertices,
            indices,
            index_count: mesh.index_count(),
            material,
        }
    }
}

pub struct TemplateParams {
    pub label: &'static str,
    pub albedo_factor: Vec4,
    pub metallic: f32,
    pub roughness: f32,
}

// --- View ---------------------------------------------------------------

pub struct LumberCampView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,

    material: PbrMaterial,
    templates: HashMap<RenderId, MeshTemplate>,
    buckets: InstanceBuckets<RenderId, MeshInstanceAttribs>,

    terrain: TerrainRenderer,
    terrain_material: TerrainMaterial,
    terrain_solid: TerrainMaterialInstance,

    /// Object currently under the cursor, resolved from the GPU hit-ID
    /// readback (1–3 frame lag is invisible at hover-highlight latency).
    /// The render walk only compares this for equality, so a stale id
    /// after an object removal just produces a one-frame no-highlight.
    hovered: Option<WorldObjectRef>,

    /// Per-kind submodule for pawns: owns the carried-log template,
    /// tick-boundary interpolation snapshots, and the idle-bob clock.
    pawn: PawnRenderer,
    /// Per-kind submodule for trees: owns the designation-marker template
    /// and its per-frame scratch.
    tree: TreeRenderer,
}

impl View for LumberCampView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — lumber camp (skeleton)",
        clear_colour: wgpu::Color {
            r: 0.45,
            g: 0.65,
            b: 0.95,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
    };

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(&renderer.device);
        // Park the rig over the centre of the map, tilted down enough to see
        // the whole zone at a comfortable pitch.
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.distance = 22.0;
        rig.pitch = 55.0_f32.to_radians();

        let samplers = SamplerRegistry::new(&renderer.device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());

        // 1×1 white sRGB texture — albedo is driven entirely by the per-template
        // `albedo_factor`. Swapping in real textures later means only changing
        // the texture handed to each template.
        let albedo = Texture::from_rgba8(
            renderer,
            "lumber-camp white",
            1,
            1,
            &[255, 255, 255, 255],
            true,
        );

        // Every body template comes from its per-kind submodule, registered
        // here in the central map so the bucket draw loop renders all kinds
        // uniformly. Adding a kind is one new line.
        let mut templates = HashMap::new();
        templates.insert(
            RenderId::Pawn,
            pawn::new_body_template(renderer, &material, &samplers, &albedo),
        );
        templates.insert(
            RenderId::Tree,
            tree::new_body_template(renderer, &material, &samplers, &albedo),
        );
        templates.insert(
            RenderId::Stockpile,
            stockpile::new_body_template(renderer, &material, &samplers, &albedo),
        );

        let mut buckets = InstanceBuckets::<RenderId, MeshInstanceAttribs>::new(
            "lumber-camp instances",
            MAX_INSTANCES_PER_TEMPLATE,
        );
        for &kind in &[RenderId::Pawn, RenderId::Tree, RenderId::Stockpile] {
            buckets.register(&renderer.device, kind);
        }

        let terrain_material = TerrainMaterial::new(renderer, camera_binding.layout());
        let terrain_solid = terrain_material.create_instance(renderer, TERRAIN_TINT);

        let pawn_renderer = PawnRenderer::new(renderer, &material, &samplers, &albedo);
        let tree_renderer = TreeRenderer::new(renderer, &material, &samplers, &albedo);

        Self {
            camera,
            camera_binding,
            rig,
            material,
            templates,
            buckets,
            terrain: TerrainRenderer::new(),
            terrain_material,
            terrain_solid,
            hovered: None,
            pawn: pawn_renderer,
            tree: tree_renderer,
        }
    }

    fn update(&mut self, sim: &Game, ctx: &mut EngineCtx, dt: Duration) {
        // Wall-clock rig integration so held-WASD pan and zoom keep working
        // independently of sim speed.
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);
        // Pawn submodule rolls over its tick-boundary interpolation
        // snapshots; no-op on frames where the sim didn't tick.
        self.pawn.update(sim, ctx);
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.zone)
    }

    fn extract_environment(&self, _: &Game, _: ZoneId) -> ViewEnvironment {
        // Fixed afternoon sun. The slow day/night cycle deliverable in #58
        // replaces this with a time-of-day-driven extract once SimEnvironment
        // is plumbed in.
        ViewEnvironment {
            sun_direction: Vec3::new(0.45, 0.35, 0.8).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85) * 2.2,
            ambient: Vec3::new(0.30, 0.32, 0.38),
            sky_color: Vec3::new(0.45, 0.65, 0.95),
        }
    }

    fn render(
        &mut self,
        sim: &Game,
        alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }
        self.camera.far = 200.0;
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Lazy first-frame terrain upload — init() has no sim handle.
        let zone = sim.zones.get(sim.zone).expect("main zone");
        if self.terrain.is_empty() {
            let mesher = FlatTopsMesher {
                tile_size: TILE_SIZE,
                height_unit: HEIGHT_UNIT,
                ..FlatTopsMesher::new()
            };
            self.terrain.rebuild_all(renderer, zone.terrain(), &mesher);
        }

        // Resolve the current hover from the GPU readback. `hit_id_hover`
        // can lag rendering by 1–3 frames; for hover-highlight that's
        // invisible. Terrain hits and "no hit" both clear the object hover.
        self.hovered = match renderer.hit_id_hover() {
            Some(HitTarget::Object { zone, id }) => Some(WorldObjectRef { zone, id }),
            _ => None,
        };

        // Single fused walk over every sim object with a RenderId, pushing
        // one per-instance attribs to the matching bucket. Per-kind details
        // (pawn interpolation + idle bob + log emission) are delegated to
        // the kind's submodule but invoked inline so we keep a single pass
        // over the zone — see the cache/i-cache rationale in pawn.rs.
        self.buckets.begin_frame();
        self.pawn.begin_frame();
        self.tree.begin_frame();
        for (zone_id, zone) in sim.zones.iter() {
            for (id, transform) in zone.iter() {
                let Some(&render_id) = zone.components().get::<RenderId>(id) else {
                    continue;
                };
                let position = if render_id == RenderId::Pawn {
                    self.pawn.position_for(zone, id, transform.position, alpha)
                } else {
                    transform.position
                };
                let model = Mat4::from_rotation_translation(transform.rotation, position);
                let hit_id = renderer.reserve_object(zone_id, id);
                let tint = if self.hovered == Some(WorldObjectRef { zone: zone_id, id }) {
                    HOVER_TINT
                } else {
                    Vec4::ONE
                };
                self.buckets.push(
                    render_id,
                    MeshInstanceAttribs::new(model, tint).with_hit_id(hit_id),
                );
                match render_id {
                    RenderId::Pawn => self.pawn.push_log_if_carrying(zone, id, model),
                    RenderId::Tree => self.tree.push_marker_if_designated(zone, id, transform),
                    RenderId::Stockpile => {}
                }
            }
        }
        self.buckets.upload(&renderer.queue);
        self.pawn.upload_logs(&renderer.queue);
        self.tree.upload_markers(&renderer.queue);

        // Terrain first so opaque ground is in the depth buffer before meshes
        // draw on top of it. Same camera + scene env bindings serve both.
        // No liquids in the skeleton — skip the transparent pass.
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.zone, &self.terrain_solid);

        // Mesh objects: one indexed-instanced call per template, swapping
        // material bind group + vertex/index buffers each time.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        for (render_id, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(template) = self.templates.get(&render_id) else {
                continue;
            };
            pass.set_bind_group(2, template.material.bind_group(), &[]);
            pass.set_vertex_buffer(0, template.vertices.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(template.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..template.index_count, 0, 0..count);
        }

        // Per-kind ancillary draws — same pipeline as the main meshes;
        // each submodule binds its own template + scratch buffer and
        // issues the indexed-instanced draw.
        self.tree.draw_markers(pass);
        self.pawn.draw_logs(pass);
    }

    fn input(&mut self, sim: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
        self.rig.handle_event(event);
        match event {
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Escape)) =>
            {
                ctx.event_loop.exit();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => tree::toggle_designation_under_cursor(sim, self.hovered),
            _ => {}
        }
    }

    fn game_ui(&mut self, sim: &mut Game, ctx: &mut EngineCtx) {
        // Top-left status panel: wood progress + countdown. Always present
        // so the player sees the goal even after winning.
        let wood = sim.wood_count();
        let remaining = (TIME_LIMIT_SECS - sim.elapsed).max(0.0);
        yakui::pad(yakui::widgets::Pad::all(16.0), || {
            yakui::align(yakui::Alignment::TOP_LEFT, || {
                yakui::colored_box_container(yakui::Color::rgba(20, 24, 32, 220), || {
                    yakui::pad(yakui::widgets::Pad::all(12.0), || {
                        yakui::column(|| {
                            yakui::label(format!("Wood: {wood} / {WOOD_GOAL}"));
                            yakui::label(format!("Time: {}", format_clock(remaining)));
                            if yakui::button("quit").clicked {
                                ctx.event_loop.exit();
                            }
                        });
                    });
                });
            });
        });

        // Centre banner on game-over. Colour-coded so win and loss read
        // distinctly even before the eye parses the text.
        let banner = match sim.state {
            GameState::Won => Some(("VICTORY", yakui::Color::rgba(40, 100, 50, 240))),
            GameState::Lost => Some(("OUT OF TIME", yakui::Color::rgba(120, 40, 40, 240))),
            GameState::Playing => None,
        };
        if let Some((text, bg)) = banner {
            yakui::align(yakui::Alignment::CENTER, || {
                yakui::colored_box_container(bg, || {
                    yakui::pad(yakui::widgets::Pad::all(28.0), || {
                        yakui::column(|| {
                            yakui::label(text);
                            yakui::label("Esc to quit");
                        });
                    });
                });
            });
        }
    }
}

/// Format `seconds` as `M:SS` for the HUD countdown.
fn format_clock(seconds: f32) -> String {
    let total = seconds.max(0.0) as u32;
    format!("{}:{:02}", total / 60, total % 60)
}
