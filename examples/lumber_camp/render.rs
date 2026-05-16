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
//! cone floating above its apex, and every [`Carrying`] pawn gets a brown
//! log riding on their shoulders (rotates with the pawn for free, since
//! the log's model matrix is `pawn_model * log_local`).

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, HitTarget, InstanceBuckets,
    MeshInstanceAttribs, OrbitRig, PbrMaterial, PbrMaterialInstance, PbrMaterialParams,
    PosNormalUv, PrimitiveMesh, Renderer, SamplerKind, SamplerRegistry, TerrainMaterial,
    TerrainMaterialInstance, TerrainRenderer, Texture, View, ViewConfig, ViewEnvironment,
    WorldObjectId, WorldObjectRef, ZoneId, wgpu, winit,
};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::sim::{Carrying, Designated, Game, HEIGHT_UNIT, Move, RenderId, TILE_SIZE};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES_PER_TEMPLATE: u32 = 64;
const TERRAIN_TINT: Vec4 = Vec4::new(0.55, 0.65, 0.42, 1.0); // grass green
/// Per-instance multiplier applied to the hovered object's albedo. Boosted
/// above one so the highlight overdrives the lit colour, warm-gold biased
/// so the signal reads as "selectable" against the neutral scene palette.
const HOVER_TINT: Vec4 = Vec4::new(1.8, 1.6, 0.8, 1.0);
/// Upper bound on simultaneously-designated trees rendered in one frame.
/// Way above the playable count; sized for the marker instance buffer only.
const MAX_MARKERS: u32 = 64;
/// Half the tree's vertical extent — used to compute the world-space apex
/// of a tree whose transform position is its centre. Mirrors the cone height
/// used to build the tree mesh.
const TREE_HALF_HEIGHT: f32 = 1.0;
/// Vertical air gap between the tree apex and the marker apex. Keeps the
/// marker from kissing the canopy at this camera distance.
const MARKER_GAP: f32 = 0.12;
/// Half the marker cone's height. Matches the marker mesh built in `init`.
const MARKER_HALF_HEIGHT: f32 = 0.175;
/// Vertical amplitude of the idle bob applied to pawns without a Move.
/// A few centimetres reads as breathing without looking like a glitch.
const IDLE_BOB_AMPLITUDE: f32 = 0.035;
/// Idle-bob frequency in Hz. Slow enough to feel like a breath rather
/// than a hop; wall-clock driven so paused pawns still breathe.
const IDLE_BOB_HZ: f32 = 1.4;
/// Upper bound on simultaneously-carried logs (= number of pawns in
/// flight to the stockpile). Sized for the log instance buffer.
const MAX_LOGS: u32 = 32;

// --- Per-template GPU resources -----------------------------------------

/// Mesh buffers + the PBR material instance to draw them with. One per
/// [`RenderId`] kind; bound together because the draw loop always swaps
/// both at once.
struct MeshTemplate {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: PbrMaterialInstance,
}

impl MeshTemplate {
    fn new(
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

struct TemplateParams {
    label: &'static str,
    albedo_factor: Vec4,
    metallic: f32,
    roughness: f32,
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

    /// Marker template (small red cone) drawn above every Designated tree.
    /// One mesh, one material, one instance buffer — `InstanceBuckets`
    /// would be overkill for a single bucket; inline the per-frame upload.
    marker_template: MeshTemplate,
    marker_buffer: wgpu::Buffer,
    marker_scratch: Vec<MeshInstanceAttribs>,

    /// Log template (small wood-brown cylinder) drawn across the shoulders
    /// of every pawn with a `Carrying` component. Pushed inline during the
    /// main pawn-render walk so the log inherits the pawn's interpolated
    /// position + facing for free.
    log_template: MeshTemplate,
    log_buffer: wgpu::Buffer,
    log_scratch: Vec<MeshInstanceAttribs>,

    /// Pawn positions captured at the previous tick boundary. With
    /// [`pawn_curr`](Self::pawn_curr) and the current tick's `alpha`, the
    /// render path lerps pawn world-positions for smooth motion at any
    /// render rate. Non-pawn objects don't move; they draw from the live
    /// sim transform.
    pawn_prev: HashMap<WorldObjectId, Vec3>,
    /// Pawn positions captured at the latest tick boundary.
    pawn_curr: HashMap<WorldObjectId, Vec3>,
    /// `SimClock::total_ticks` at the last `update` that took a snapshot;
    /// snapshots only roll over when this changes, so paused frames keep
    /// `prev` and `curr` matched and the lerp pins to the sim position.
    last_seen_tick: u64,

    /// Wall-clock origin used to drive view-side animation (currently the
    /// idle-bob phase). Wall-clock — not sim time — so the bob keeps
    /// breathing while the sim is paused.
    started: Instant,
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

        // Primitive meshes sized for the world: capsule pawns ~1.6 m tall,
        // cone trees ~2 m tall, 1 m³ stockpile cube. Z-up: every primitive is
        // centred on the origin, and the sim positions place the centre at
        // height/2 so the base sits on the ground. The pawn carries a small
        // "satchel" cube on its local +X side so per-instance rotation is
        // visible on an otherwise rotationally-symmetric capsule.
        let pawn_mesh = pawn_mesh_with_satchel();
        let tree_mesh = PrimitiveMesh::cone(0.60, 2.0, 16, true);
        let stockpile_mesh = PrimitiveMesh::cube(Vec3::ONE);

        let mut templates = HashMap::new();
        templates.insert(
            RenderId::Pawn,
            MeshTemplate::new(
                renderer,
                &material,
                &samplers,
                &albedo,
                &pawn_mesh,
                TemplateParams {
                    label: "lumber-camp pawn",
                    albedo_factor: Vec4::new(0.95, 0.70, 0.55, 1.0), // skin-warm
                    metallic: 0.0,
                    roughness: 0.70,
                },
            ),
        );
        templates.insert(
            RenderId::Tree,
            MeshTemplate::new(
                renderer,
                &material,
                &samplers,
                &albedo,
                &tree_mesh,
                TemplateParams {
                    label: "lumber-camp tree",
                    albedo_factor: Vec4::new(0.20, 0.50, 0.18, 1.0), // foliage green
                    metallic: 0.0,
                    roughness: 0.95,
                },
            ),
        );
        templates.insert(
            RenderId::Stockpile,
            MeshTemplate::new(
                renderer,
                &material,
                &samplers,
                &albedo,
                &stockpile_mesh,
                TemplateParams {
                    label: "lumber-camp stockpile",
                    albedo_factor: Vec4::new(0.55, 0.32, 0.18, 1.0), // wood brown
                    metallic: 0.0,
                    roughness: 0.85,
                },
            ),
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

        // Designation marker: small cone, oriented apex-down by the per-
        // instance model matrix at render time. Bright red albedo so it
        // reads against the green canopy without needing a per-frame tint.
        let marker_mesh = PrimitiveMesh::cone(0.18, 0.35, 12, true);
        let marker_template = MeshTemplate::new(
            renderer,
            &material,
            &samplers,
            &albedo,
            &marker_mesh,
            TemplateParams {
                label: "lumber-camp designation marker",
                albedo_factor: Vec4::new(1.0, 0.15, 0.15, 1.0), // bright red
                metallic: 0.0,
                roughness: 0.55,
            },
        );
        let marker_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lumber-camp marker instances"),
            size: u64::from(MAX_MARKERS) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Carried log: short cylinder oriented horizontal in the pawn's
        // local frame at render time, riding on the pawn's model matrix.
        // Brown so it reads as wood against the body's skin tone.
        let log_mesh = PrimitiveMesh::cylinder(0.07, 0.6, 12, true);
        let log_template = MeshTemplate::new(
            renderer,
            &material,
            &samplers,
            &albedo,
            &log_mesh,
            TemplateParams {
                label: "lumber-camp carried log",
                albedo_factor: Vec4::new(0.42, 0.27, 0.16, 1.0), // wood brown
                metallic: 0.0,
                roughness: 0.85,
            },
        );
        let log_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lumber-camp log instances"),
            size: u64::from(MAX_LOGS) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            marker_template,
            marker_buffer,
            marker_scratch: Vec::with_capacity(MAX_MARKERS as usize),
            log_template,
            log_buffer,
            log_scratch: Vec::with_capacity(MAX_LOGS as usize),
            pawn_prev: HashMap::new(),
            pawn_curr: HashMap::new(),
            last_seen_tick: 0,
            started: Instant::now(),
        }
    }

    fn update(&mut self, sim: &Game, ctx: &mut EngineCtx, dt: Duration) {
        // Wall-clock rig integration so held-WASD pan and zoom keep working
        // independently of sim speed.
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);

        // Tick-boundary pawn snapshot. The standard fixed-tick interpolation
        // pattern: previous tick's positions are the lerp source, this
        // tick's positions are the destination, and `alpha` in the next
        // render call picks where between them we draw. Snapshot only when
        // the tick counter has actually advanced, so paused / between-tick
        // frames don't churn the maps.
        let now_tick = ctx.clock.total_ticks();
        if now_tick != self.last_seen_tick {
            self.pawn_prev = std::mem::take(&mut self.pawn_curr);
            if let Some(zone) = sim.zones.get(sim.zone) {
                for (id, transform) in zone.iter() {
                    if zone.components().get::<RenderId>(id) == Some(&RenderId::Pawn) {
                        self.pawn_curr.insert(id, transform.position);
                    }
                }
            }
            self.last_seen_tick = now_tick;
        }
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

        // Walk every sim object with a RenderId and push a per-instance
        // attribs to the matching bucket. Hit IDs come from the engine so the
        // ID buffer resolves cursor hits back to the parent WorldObjectId.
        // Pawn positions lerp between the two most recent tick-boundary
        // snapshots using `alpha`; idle pawns get a wall-clock sin-bob on
        // top of that. Other objects don't move, so they draw from the live
        // sim transform. Carrying pawns also push a log instance whose
        // model matrix is `pawn_model * log_local` — log inherits position,
        // facing, and any future per-pawn animation for free.
        let bob_phase = self.started.elapsed().as_secs_f32() * IDLE_BOB_HZ * TAU;
        let log_local = Mat4::from_rotation_translation(
            Quat::from_rotation_x(PI / 2.0),
            Vec3::new(0.0, 0.0, 0.95),
        );
        self.buckets.begin_frame();
        self.log_scratch.clear();
        for (zone_id, zone) in sim.zones.iter() {
            for (id, transform) in zone.iter() {
                let Some(&render_id) = zone.components().get::<RenderId>(id) else {
                    continue;
                };
                let position = if render_id == RenderId::Pawn {
                    let prev = self
                        .pawn_prev
                        .get(&id)
                        .copied()
                        .unwrap_or(transform.position);
                    let curr = self
                        .pawn_curr
                        .get(&id)
                        .copied()
                        .unwrap_or(transform.position);
                    let mut pos = prev.lerp(curr, alpha);
                    if zone.components().get::<Move>(id).is_none() {
                        pos.z += bob_phase.sin() * IDLE_BOB_AMPLITUDE;
                    }
                    pos
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

                if render_id == RenderId::Pawn
                    && zone.components().get::<Carrying>(id).is_some()
                    && self.log_scratch.len() < MAX_LOGS as usize
                {
                    let log_model = model * log_local;
                    self.log_scratch
                        .push(MeshInstanceAttribs::new(log_model, Vec4::ONE));
                }
            }
        }
        self.buckets.upload(&renderer.queue);
        if !self.log_scratch.is_empty() {
            renderer.queue.write_buffer(
                &self.log_buffer,
                0,
                bytemuck::cast_slice(&self.log_scratch),
            );
        }

        // Terrain first so opaque ground is in the depth buffer before meshes
        // draw on top of it. Same camera + scene env bindings serve both.
        // No liquids in the skeleton — skip the transparent pass.
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.zone, &self.terrain_solid);

        // Designation markers — one per tree with the `Designated` component.
        // Rotated apex-down so the cone points at the tree it marks; floats
        // a few cm above the tree's apex (tree height/2 above its centre,
        // plus a small visible gap). Markers don't reserve hit IDs — the
        // tree underneath is the click target, not the marker.
        self.marker_scratch.clear();
        for (_zone_id, zone) in sim.zones.iter() {
            for (id, _) in zone.components().iter::<Designated>() {
                let Some(tree) = zone.get(id) else { continue };
                let tree_apex_z = tree.position.z + TREE_HALF_HEIGHT;
                let marker_centre = Vec3::new(
                    tree.position.x,
                    tree.position.y,
                    tree_apex_z + MARKER_GAP + MARKER_HALF_HEIGHT,
                );
                let model =
                    Mat4::from_rotation_translation(Quat::from_rotation_x(PI), marker_centre);
                if self.marker_scratch.len() < MAX_MARKERS as usize {
                    self.marker_scratch
                        .push(MeshInstanceAttribs::new(model, Vec4::ONE));
                }
            }
        }
        if !self.marker_scratch.is_empty() {
            renderer.queue.write_buffer(
                &self.marker_buffer,
                0,
                bytemuck::cast_slice(&self.marker_scratch),
            );
        }

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

        // Markers and logs — same pipeline as the main meshes, just swap
        // the template's mesh + material bind group and draw the per-frame
        // scratch range.
        if !self.marker_scratch.is_empty() {
            pass.set_bind_group(2, self.marker_template.material.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.marker_template.vertices.slice(..));
            pass.set_vertex_buffer(1, self.marker_buffer.slice(..));
            pass.set_index_buffer(
                self.marker_template.indices.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(
                0..self.marker_template.index_count,
                0,
                0..self.marker_scratch.len() as u32,
            );
        }
        if !self.log_scratch.is_empty() {
            pass.set_bind_group(2, self.log_template.material.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.log_template.vertices.slice(..));
            pass.set_vertex_buffer(1, self.log_buffer.slice(..));
            pass.set_index_buffer(
                self.log_template.indices.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed(
                0..self.log_template.index_count,
                0,
                0..self.log_scratch.len() as u32,
            );
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
            } => toggle_designation_under_cursor(sim, self.hovered),
            _ => {}
        }
    }
}

/// Toggle a [`Designated`] component on whichever sim object is currently
/// under the cursor — but only if it's a tree. Pawns and the stockpile
/// click through (left-click on those will mean "select" in a later slice).
fn toggle_designation_under_cursor(sim: &mut Game, hovered: Option<WorldObjectRef>) {
    let Some(WorldObjectRef { zone, id }) = hovered else {
        return;
    };
    let Some(zone) = sim.zones.get_mut(zone) else {
        return;
    };
    // A readback id can be stale across object removal; component lookup on a
    // dead id returns None and we no-op.
    if !matches!(zone.components().get::<RenderId>(id), Some(RenderId::Tree)) {
        return;
    }
    if zone.components().get::<Designated>(id).is_some() {
        zone.components_mut().remove::<Designated>(id);
    } else {
        zone.components_mut().insert(id, Designated);
    }
}

/// Pawn body + a small offset cube ("satchel") on the local +X side, baked
/// into one mesh. Same material as the body, so the satchel doesn't
/// visually pop on its own — but the geometric protrusion catches the sun
/// at a different angle than the capsule surface, making the pawn's facing
/// readable as it walks around.
///
/// When the example outgrows single-mesh templates (or wants the satchel a
/// different colour), the right move is to adopt `RenderTemplate` +
/// `RenderObjectPass::for_each_alive_part`, the same multi-part / multi-
/// material path the campfire example uses.
fn pawn_mesh_with_satchel() -> PrimitiveMesh {
    let mut mesh = PrimitiveMesh::capsule(0.30, 1.6, 16, 3);
    let satchel = PrimitiveMesh::cube(Vec3::splat(0.18));
    // Sit the cube just outside the capsule wall on +X, at upper-torso
    // height, so it reads as a visible asymmetry from any camera angle.
    let offset = Vec3::new(0.34, 0.0, 0.45);
    let base = mesh.vertices.len() as u32;
    mesh.vertices
        .extend(satchel.vertices.iter().map(|v| PosNormalUv {
            position: [
                v.position[0] + offset.x,
                v.position[1] + offset.y,
                v.position[2] + offset.z,
            ],
            normal: v.normal,
            uv: v.uv,
        }));
    mesh.indices
        .extend(satchel.indices.iter().map(|&i| base + i));
    mesh
}
