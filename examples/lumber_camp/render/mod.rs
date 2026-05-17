//! View-side rendering for the lumber camp.
//!
//! Three engine [`RenderTemplate`]s — pawn (body + carried log), tree
//! (body + designation marker), stockpile (body only) — drawn through one
//! PBR pipeline against a fixed afternoon sun. Each `MeshPart` on those
//! templates resolves to a per-part [`MeshTemplate`] (mesh buffers + PBR
//! material instance) keyed by [`PartKey`]; the draw loop is one
//! indexed-instanced call per `PartKey` bucket.
//!
//! Per-frame the engine drives the walk: [`RenderObjectPass::declare_and_cull`]
//! emits one live proxy per sim object carrying a [`RenderId`], culls
//! against the camera frustum (with hysteresis), then
//! [`RenderObjectPass::update_instances`] runs once per alive proxy as
//! the *single sim→view translation seam*. That closure reads typed sim
//! components (`Carrying`, `Move`, `Designated`) and writes view-side
//! per-instance state on the proxy:
//! - `instance.mesh_parts[LOG_PART].visible` for carried logs,
//! - `instance.mesh_parts[MARKER_PART].visible` for designation markers,
//! - `instance.world_xform` overwritten with the pawn's interpolated +
//!   idle-bobbed pose, which cascades into the log part automatically
//!   via the engine's `world_xform * part.local_transform` compose.
//!
//! [`RenderObjectPass::for_each_alive_part_with_hit_id`] then does the
//! actual draw-attrib push, reserving one hit ID per parent so clicking
//! either the body or an ancillary part (carried log, designation
//! marker) resolves back to the same sim object.

mod pawn;
mod stockpile;
mod tree;

use std::collections::HashMap;
use std::time::Duration;

use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, Frustum, HitTarget, InstanceBuckets,
    LiveRenderObjects, MeshInstanceAttribs, OrbitRig, PbrMaterial, PbrMaterialInstance,
    PbrMaterialParams, PrimitiveMesh, RenderObjectPass, RenderRegistry, RenderTemplate, Renderer,
    SamplerKind, SamplerRegistry, TerrainMaterial, TerrainMaterialInstance, TerrainRenderer,
    Texture, View, ViewConfig, ViewEnvironment, WorldObjectRef, ZoneId, wgpu, winit, yakui,
};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use pawn::PawnRenderer;

use crate::sim::{
    Carrying, Designated, Game, GameState, HEIGHT_UNIT, Move, RenderId, TILE_SIZE, TIME_LIMIT_SECS,
    WOOD_GOAL,
};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES_PER_PART: u32 = 64;
/// 30 frames matches CLAUDE.md's hysteresis recommendation — enough to
/// hide pop-out at grazing camera angles around the orbit rig.
const CULL_HYSTERESIS_FRAMES: u32 = 30;
const TERRAIN_TINT: Vec4 = Vec4::new(0.55, 0.65, 0.42, 1.0); // grass green
/// Per-instance multiplier applied to the hovered object's albedo. Boosted
/// above one so the highlight overdrives the lit colour, warm-gold biased
/// so the signal reads as "selectable" against the neutral scene palette.
pub(crate) const HOVER_TINT: Vec4 = Vec4::new(1.8, 1.6, 0.8, 1.0);

// --- Per-template GPU resources ----------------------------------------

/// Names one drawable part across every render template. Both the *mesh*
/// and *material* of a [`MeshPart`] in this example resolve to the same
/// `MeshTemplate` (mesh buffers + PBR material instance bundled), so one
/// enum serves both roles in [`RenderTemplate<RenderId, PartKey, PartKey>`].
/// Adding a new visual part is a new variant + a [`MeshTemplate`] entry
/// in `mesh_templates`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PartKey {
    PawnBody,
    Log,
    TreeBody,
    Marker,
    Stockpile,
}

impl PartKey {
    /// Whether this part receives the per-frame hover tint. Body parts
    /// do (the readable "I'm hovering this object" signal sits on the
    /// silhouette people read first); ancillary parts (carried log,
    /// designation marker) keep their own constant albedo so the marker
    /// stays red and the log stays brown.
    fn is_body(self) -> bool {
        matches!(self, Self::PawnBody | Self::TreeBody | Self::Stockpile)
    }
}

/// Mesh buffers + the PBR material instance to draw them with. One per
/// [`PartKey`]; bound together because the draw loop always swaps
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

/// `RenderRegistry` generics for this view — pinned once to keep the
/// `LumberCampView` field readable.
type Templates = RenderRegistry<RenderId, PartKey, PartKey>;

// --- View ---------------------------------------------------------------

pub struct LumberCampView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,

    material: PbrMaterial,
    /// GPU bundle per drawable part. Looked up by the draw loop after the
    /// per-part bucket has been filled by the engine walk.
    mesh_templates: HashMap<PartKey, MeshTemplate>,
    /// Engine-side render-object registry: maps each [`RenderId`] to a
    /// [`RenderTemplate`] describing which parts make it up and where
    /// each part sits relative to the object's world transform.
    templates: Templates,
    /// Live render-object proxies with cull hysteresis. Owns the
    /// view-side per-instance state (`world_xform`, per-part visibility)
    /// that the per-instance update closure writes each frame.
    live_objects: LiveRenderObjects<RenderId>,
    /// Per-part instance-attrib buckets. The extract closure pushes one
    /// [`MeshInstanceAttribs`] per visible part; the draw loop iterates
    /// non-empty buckets and issues one indexed-instanced call each.
    buckets: InstanceBuckets<PartKey, MeshInstanceAttribs>,

    terrain: TerrainRenderer,
    terrain_material: TerrainMaterial,
    terrain_solid: TerrainMaterialInstance,

    /// Object currently under the cursor, resolved from the GPU hit-ID
    /// readback (1–3 frame lag is invisible at hover-highlight latency).
    /// Stored as `WorldObjectRef` so it lines up with the parent that
    /// the engine walk receives in callbacks.
    hovered: Option<WorldObjectRef>,

    /// Per-kind view-state for pawns: interpolation snapshots + idle-bob
    /// clock. Read by the per-instance update closure to overwrite
    /// `instance.world_xform`. Kept as a side-table on the view (rather
    /// than on `LiveRenderObject`) because there's no view-extensible
    /// state slot on the engine proxy yet — see #65 Open Question 2.
    pawn: PawnRenderer,
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

        // 1×1 white sRGB texture — albedo is driven entirely by the per-part
        // `albedo_factor`. Swapping in real textures later means only changing
        // the texture handed to each `MeshTemplate`.
        let albedo = Texture::from_rgba8(
            renderer,
            "lumber-camp white",
            1,
            1,
            &[255, 255, 255, 255],
            true,
        );

        // GPU resources per drawable part. Each `PartKey` resolves to one
        // `MeshTemplate` here; multiple `MeshPart`s in different render
        // templates may reference the same `PartKey` (none do today, but
        // the indirection is cheap).
        let mut mesh_templates = HashMap::new();
        mesh_templates.insert(
            PartKey::PawnBody,
            pawn::new_body_template(renderer, &material, &samplers, &albedo),
        );
        mesh_templates.insert(
            PartKey::Log,
            pawn::new_log_template(renderer, &material, &samplers, &albedo),
        );
        mesh_templates.insert(
            PartKey::TreeBody,
            tree::new_body_template(renderer, &material, &samplers, &albedo),
        );
        mesh_templates.insert(
            PartKey::Marker,
            tree::new_marker_template(renderer, &material, &samplers, &albedo),
        );
        mesh_templates.insert(
            PartKey::Stockpile,
            stockpile::new_body_template(renderer, &material, &samplers, &albedo),
        );

        // Engine render templates: which parts make up each `RenderId`,
        // where each part sits, and the visual AABB used for frustum
        // culling. Both the `mesh` and `material` of each `MeshPart`
        // resolve to the same `PartKey` for this example.
        let mut templates: Templates = RenderRegistry::new();
        templates.register(
            RenderId::Pawn,
            RenderTemplate::new("pawn")
                .with_mesh_part(PartKey::PawnBody, PartKey::PawnBody, Mat4::IDENTITY)
                .with_mesh_part(PartKey::Log, PartKey::Log, pawn::log_local_transform())
                .with_visual_bounds(pawn::visual_bounds()),
        );
        templates.register(
            RenderId::Tree,
            RenderTemplate::new("tree")
                .with_mesh_part(PartKey::TreeBody, PartKey::TreeBody, Mat4::IDENTITY)
                .with_mesh_part(
                    PartKey::Marker,
                    PartKey::Marker,
                    tree::marker_local_transform(),
                )
                .with_visual_bounds(tree::visual_bounds()),
        );
        templates.register(
            RenderId::Stockpile,
            RenderTemplate::new("stockpile")
                .with_mesh_part(PartKey::Stockpile, PartKey::Stockpile, Mat4::IDENTITY)
                .with_visual_bounds(stockpile::visual_bounds()),
        );

        let live_objects = LiveRenderObjects::<RenderId>::new(CULL_HYSTERESIS_FRAMES);

        let mut buckets = InstanceBuckets::<PartKey, MeshInstanceAttribs>::new(
            "lumber-camp instances",
            MAX_INSTANCES_PER_PART,
        );
        for &key in &[
            PartKey::PawnBody,
            PartKey::Log,
            PartKey::TreeBody,
            PartKey::Marker,
            PartKey::Stockpile,
        ] {
            buckets.register(&renderer.device, key);
        }

        let terrain_material = TerrainMaterial::new(renderer, camera_binding.layout());
        let terrain_solid = terrain_material.create_instance(renderer, TERRAIN_TINT);

        Self {
            camera,
            camera_binding,
            rig,
            material,
            mesh_templates,
            templates,
            live_objects,
            buckets,
            terrain: TerrainRenderer::new(),
            terrain_material,
            terrain_solid,
            hovered: None,
            pawn: PawnRenderer::new(),
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

        self.pawn.begin_frame();
        self.buckets.begin_frame();

        // Phase 1: engine-driven walk + declare + cull. One proxy per sim
        // object carrying a `RenderId`, frustum-culled with hysteresis.
        let frustum = Frustum::from_view_proj(self.camera.view_proj());
        RenderObjectPass::declare_and_cull(
            &sim.zones,
            &self.templates,
            &mut self.live_objects,
            &frustum,
        );

        // Phase 1.5: the single sim→view translation seam. Reads typed
        // sim components and writes view-side state on each alive proxy.
        // Disjoint-borrows `self.pawn` (read-only) and `self.live_objects`
        // (mutable through the engine call) by binding each before the
        // closure.
        let pawn = &self.pawn;
        RenderObjectPass::update_instances(
            &sim.zones,
            &self.templates,
            &mut self.live_objects,
            |parent, rid, _slots, components, instance| match rid {
                RenderId::Pawn => {
                    // Override world_xform with the interpolated +
                    // idle-bobbed pose. Engine composes
                    // `world_xform * log.local_transform` for the log
                    // part, so the log inherits this pose automatically.
                    let live_position = instance.world_xform.w_axis.truncate();
                    let has_move = components.get::<Move>(parent.id).is_some();
                    let pos = pawn.interp_position(parent.id, live_position, alpha, has_move);
                    instance.world_xform.w_axis = Vec4::new(pos.x, pos.y, pos.z, 1.0);
                    // Carried log: visible iff the pawn has a Carrying.
                    instance.mesh_parts[pawn::LOG_PART].visible =
                        components.get::<Carrying>(parent.id).is_some();
                }
                RenderId::Tree => {
                    // Designation marker: visible iff the tree has Designated.
                    instance.mesh_parts[tree::MARKER_PART].visible =
                        components.get::<Designated>(parent.id).is_some();
                }
                RenderId::Stockpile => {
                    // No view-side decisions — defaults (all visible) stand.
                }
            },
        );

        // Phase 2: engine-driven extract. The hit-ID-aware variant reserves
        // one ID per parent so clicking the log resolves back to the pawn,
        // and clicking the marker resolves back to the tree (#56 PR 3).
        // Hover tint stays on body parts; the carried log and designation
        // marker keep their own constant albedo for legibility.
        let buckets = &mut self.buckets;
        let hovered = self.hovered;
        RenderObjectPass::for_each_alive_part_with_hit_id(
            &sim.zones,
            &self.templates,
            &self.live_objects,
            renderer,
            |parent, _rid, part, world, _slots, hit_id| {
                let tint = if hovered == Some(parent) && part.material.is_body() {
                    HOVER_TINT
                } else {
                    Vec4::ONE
                };
                buckets.push(
                    part.mesh,
                    MeshInstanceAttribs::new(world, tint).with_hit_id(hit_id),
                );
            },
        );

        self.buckets.upload(&renderer.queue);

        // Terrain first so opaque ground is in the depth buffer before meshes
        // draw on top of it. Same camera + scene env bindings serve both.
        // No liquids in the skeleton — skip the transparent pass.
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.zone, &self.terrain_solid);

        // Mesh parts: one indexed-instanced call per non-empty bucket,
        // swapping material bind group + vertex/index buffers each time.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        for (part_key, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(template) = self.mesh_templates.get(&part_key) else {
                continue;
            };
            pass.set_bind_group(2, template.material.bind_group(), &[]);
            pass.set_vertex_buffer(0, template.vertices.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(template.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..template.index_count, 0, 0..count);
        }
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
