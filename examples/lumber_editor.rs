//! Lumber editor — single-item kind viewer.
//!
//! Pick a kind from the egui side panel; the chosen kind is placed at the
//! origin and rendered with its real glb + texture streamed through the
//! [`AssetServer`]. The orbit rig lets you rotate around the displayed item;
//! the camera auto-frames each kind's `bounds_min`/`bounds_max` AABB on
//! selection so a `<1 m` pawn and a 6 m building both show at a comfortable
//! distance.
//!
//! MVP scope is **select + view only** — no property editing, no save. This
//! also doubles as the minimal reference for "how do I build a view that
//! reads kinds and streams their assets" without the lumber_camp scaffolding
//! (terrain, hit-testing, shadow cascades, yakui game UI).
//!
//! ## Layout
//!
//! - `Game` (sim) owns one zone with one [`WorldTransform`] at the origin.
//!   Its sole sim mutation is `SelectKind(KindId)`, which swaps the
//!   [`KindId`] component on the single object.
//! - `LumberEditorView` mirrors lumber_camp's kind → template pattern: walk
//!   [`Definitions`] at init, build one [`RenderTemplate`] per kind that has
//!   a `render` block, and dispatch by `KindId` at draw time.
//!
//! Controls:
//! - Click a kind in the left panel — swap the displayed item, re-frame the camera.
//! - Right-click drag — rotate the camera (yaw + pitch).
//! - Scroll wheel — zoom.
//! - W / A / S / D — pan the focal point.
//! - Esc — quit.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{Definitions, FsSource, KindId, Vfs, VfsPath};
use currawong::glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use currawong::{
    AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, Facing, Frustum, Handle,
    InstanceBuckets, MaterialId, MaterialRegistry, MeshDraw, MeshInstanceAttribs, MeshTemplate,
    OrbitRig, PbrAtlasMaterial, PbrAtlasMaterialInstance, PbrAtlasMaterialParams,
    PbrAtlasMaterials, PbrMaterial, PbrMaterialInstance, PbrMaterialParams, PrimitiveMesh,
    RenderObjectTraversal, RenderProxies, RenderRegistry, RenderSpec, RenderTemplate, Renderer,
    SamplerKind, SamplerRegistry, ShadowMeshPipeline, SimPos, SimUnit, Simulation, SunCascades,
    Texture, TextureColorSpace, View, ViewConfig, ViewEnvironment, WorldObjectId, WorldTransform,
    Zone, ZoneId, Zones, egui, pollster, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- VFS ---------------------------------------------------------------

/// Mount the repo's `assets/` directory as a fresh [`Vfs`]. Called twice at
/// startup — once by `main` for the sim's [`Definitions`], once by the view
/// for the [`AssetServer`] — matching the lumber_camp convention.
fn lumber_editor_vfs() -> Vfs {
    let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let mut vfs = Vfs::new();
    vfs.mount(FsSource::new(assets_root));
    vfs
}

// --- Sim ---------------------------------------------------------------

/// Sole sim mutation: pick which kind the single displayed object renders as.
#[derive(Debug, Clone)]
enum Command {
    SelectKind(KindId),
}

struct Game {
    zones: Zones,
    zone: ZoneId,
    subject: WorldObjectId,
    /// Sorted list of every kind that has a `render` block — the source for
    /// the egui kind list. Sim-side because the UI reads it via `&Sim`.
    available: Vec<KindId>,
    /// Cached `RenderSpec` per kind for camera auto-framing. Cheap (a few
    /// dozen entries max in any reasonable kinds folder); avoids re-parsing
    /// the def on every selection.
    render_specs: HashMap<KindId, RenderSpec>,
}

impl Game {
    fn new(defs: Definitions) -> Self {
        let mut available: Vec<KindId> = Vec::new();
        let mut render_specs: HashMap<KindId, RenderSpec> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            match RenderSpec::from_def(def) {
                Ok(spec) => {
                    available.push(kind_id.clone());
                    render_specs.insert(kind_id.clone(), spec);
                }
                Err(e) => {
                    eprintln!("lumber_editor: skipping {kind_id}: {e}");
                }
            }
        }
        available.sort_by(|a, b| a.as_str().cmp(b.as_str()));

        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");
        let subject = zone.insert(WorldTransform {
            position: SimPos::new(SimUnit::ZERO, SimUnit::ZERO, SimUnit::ZERO),
            facing: Facing::ZERO,
        });
        if let Some(first) = available.first() {
            zone.components_mut().insert(subject, first.clone());
        }

        Self {
            zones,
            zone: zone_id,
            subject,
            available,
            render_specs,
        }
    }
}

impl Simulation for Game {
    type Command = Command;

    fn tick(&mut self, _dt: Duration) {}

    fn apply_command(&mut self, cmd: &Command) {
        match cmd {
            Command::SelectKind(kind) => {
                if let Some(zone) = self.zones.get_mut(self.zone) {
                    zone.components_mut().insert(self.subject, kind.clone());
                }
            }
        }
    }
}

// --- View --------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Generous enough for any single-item view; only one part draws at a time.
const MAX_INSTANCES_PER_PART: u32 = 4;

/// `PartKey` collapses to `KindId`: each kind is exactly one body part.
type Templates = RenderRegistry<KindId, KindId, KindId>;

struct LumberEditorView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,

    material: PbrMaterial,
    atlas_material: PbrAtlasMaterial,
    atlas_materials: MaterialRegistry<PbrAtlasMaterialInstance>,
    samplers: SamplerRegistry,
    asset_server: AssetServer,

    mesh_templates: HashMap<KindId, MeshTemplate<PbrMaterialInstance>>,
    templates: Templates,
    proxies: RenderProxies<KindId>,
    buckets: InstanceBuckets<KindId, MeshInstanceAttribs>,

    /// Depth-only pipeline for the four cascade shadow passes per frame.
    /// Shares the canonical `PosNormalUv` + `MeshInstanceAttribs` layout, so
    /// the same instance buckets we draw in `render` are re-bound under the
    /// depth-only pipeline.
    shadow_pipeline: ShadowMeshPipeline,

    /// Static checkerboard ground plane that catches the kind's shadow.
    /// Single fixed-instance draw issued at the top of `render`; not part of
    /// the proxy/template pipeline because it isn't tied to a sim object.
    ground: GroundPlane,

    /// Previous frame's selection; auto-frame fires when this differs from
    /// the sim's current `KindId` component on `subject`.
    last_selected: Option<KindId>,
}

/// GPU resources for the editor's static checkerboard floor. One quad, one
/// instance, one PBR material — sized large enough to fill the camera for
/// every kind, so its model matrix is identity and never updates.
struct GroundPlane {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    material: PbrMaterialInstance,
}

impl View for LumberEditorView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — lumber editor",
        clear_colour: wgpu::Color {
            r: 0.18,
            g: 0.20,
            b: 0.24,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
        // Shadows make the displayed kind read as a real object sitting on
        // the ground plane. Single-subject scene; one 2k cascade map is
        // plenty.
        shadow_map_resolution: Some(2048),
    };

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera {
            far: 80.0,
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(&renderer.device);

        // Placeholder rig state; the first auto-frame replaces these with
        // values derived from the initially-selected kind's bounds.
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.distance = 3.0;
        rig.pitch = 30.0_f32.to_radians();

        let samplers = SamplerRegistry::new(&renderer.device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());
        let atlas_material = PbrAtlasMaterial::new(renderer, camera_binding.layout());

        // View-side VFS, independent of main's — same on-disk content,
        // separate caches. Matches the lumber_camp convention.
        let vfs = Arc::new(lumber_editor_vfs());
        let asset_server = AssetServer::new(renderer, vfs.clone());

        // Register the lumber-camp building's atlas material instance. The
        // glb names its slot "Lumber" which resolves through `MaterialRegistry`
        // as `gltf:lumber`. Other kinds whose glb doesn't name an atlas slot
        // simply fall through to the streamed PBR material per kind.
        let albedo_handle = asset_server.texture(
            VfsPath::new("lumber/gradient_atlas.png").expect("valid path"),
            TextureColorSpace::Srgb,
        );
        let mre_handle = asset_server.texture(
            VfsPath::new("lumber/mre_atlas.png").expect("valid path"),
            TextureColorSpace::Linear,
        );
        let lumber_instance = atlas_material.create_instance(
            renderer,
            &samplers,
            &asset_server,
            PbrAtlasMaterialParams {
                albedo: albedo_handle,
                mre: mre_handle,
                sampler: SamplerKind::NearestClamp,
            },
        );
        let mut atlas_materials = MaterialRegistry::new();
        atlas_materials.register(
            MaterialId::new("gltf:lumber").expect("valid id"),
            lumber_instance,
        );

        // Re-parse the defs view-side to build the per-kind templates.
        // Failure here would be a build-pipeline divergence (the sim
        // already validated them), not a runtime condition.
        let defs = pollster::block_on(Definitions::load(
            &vfs,
            &VfsPath::new("kinds").expect("valid VFS path"),
        ))
        .expect("view-side definitions load");

        let mut mesh_templates: HashMap<KindId, MeshTemplate<PbrMaterialInstance>> = HashMap::new();
        let mut templates: Templates = RenderRegistry::new();

        for (kind_id, def) in defs.iter() {
            let spec = match RenderSpec::from_def(def) {
                Ok(spec) => spec,
                Err(_) => continue,
            };
            let body =
                material.streamed_template(renderer, &samplers, &asset_server, kind_id, &spec);
            let bounds = body.visual_bounds;
            mesh_templates.insert(kind_id.clone(), body);
            let template = RenderTemplate::new(kind_id.as_str())
                .with_mesh_part(kind_id.clone(), kind_id.clone(), Mat4::IDENTITY)
                .with_visual_bounds(bounds);
            templates.register(kind_id.clone(), template);
        }

        // Single live object, so hysteresis is irrelevant.
        let proxies = RenderProxies::<KindId>::new(0);
        let mut buckets = InstanceBuckets::<KindId, MeshInstanceAttribs>::new(
            "lumber-editor instances",
            MAX_INSTANCES_PER_PART,
        );
        for key in mesh_templates.keys().cloned().collect::<Vec<_>>() {
            buckets.register(&renderer.device, key);
        }

        let shadow_pipeline = ShadowMeshPipeline::new(renderer);
        let ground = build_ground_plane(renderer, &material, &samplers, &asset_server);

        Self {
            camera,
            camera_binding,
            rig,
            material,
            atlas_material,
            atlas_materials,
            samplers,
            asset_server,
            mesh_templates,
            templates,
            proxies,
            buckets,
            shadow_pipeline,
            ground,
            last_selected: None,
        }
    }

    fn input(
        &mut self,
        _sim: &Game,
        ctx: &mut EngineCtx,
        _cmds: &mut CommandQueue<Command>,
        event: &WindowEvent,
    ) {
        self.rig.handle_event(event);
        if let WindowEvent::KeyboardInput { event, .. } = event
            && event.state == ElementState::Pressed
            && event.physical_key == PhysicalKey::Code(KeyCode::Escape)
        {
            ctx.event_loop.exit();
        }
    }

    fn update(
        &mut self,
        sim: &Game,
        _ctx: &mut EngineCtx,
        _cmds: &mut CommandQueue<Command>,
        dt: Duration,
    ) {
        // Wall-clock rig integration — keeps WASD pan and scroll zoom
        // responsive regardless of sim speed (the sim doesn't tick anything
        // here anyway, but the invariant still matters).
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);
        self.maybe_auto_frame(sim);
    }

    fn ui(
        &mut self,
        sim: &Game,
        _ctx: &mut EngineCtx,
        cmds: &mut CommandQueue<Command>,
        egui_ctx: &egui::Context,
    ) {
        let current = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned();

        // egui 0.34 deprecated top-level `Panel::show` in favour of
        // `show_inside`, which needs an outer `Ui`. View callbacks receive
        // only `&Context`, and the upstream migration story for top-level
        // panel hosting in pure `&Context` callers isn't settled — the
        // deprecated path still works, so we use it.
        #[allow(deprecated)]
        egui::Panel::left("kinds")
            .resizable(false)
            .default_size(260.0)
            .show(egui_ctx, |ui| {
                ui.heading("Kinds");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for kind in &sim.available {
                        let selected = current.as_ref() == Some(kind);
                        let label = ui.selectable_label(selected, kind.as_str());
                        if label.clicked() && !selected {
                            cmds.push_now(Command::SelectKind(kind.clone()));
                        }
                    }
                });
            });
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.zone)
    }

    fn extract_environment(&self, _sim: &Game, _zone: ZoneId) -> ViewEnvironment {
        // Static lighting — an editor wants a stable presentation. Sun from
        // upper-right; cascades fitted every frame so shadows track the
        // orbit-rig camera.
        let sun_direction = Vec3::new(0.4, 0.3, 0.8).normalize();
        let splits = self.camera.cascade_split_distances(0.75);
        let matrices = self.camera.fit_shadow_cascades(sun_direction, splits);
        ViewEnvironment {
            sun_direction,
            sun_color: Vec3::splat(2.5),
            ambient: Vec3::new(0.25, 0.27, 0.32),
            sky_color: Vec3::new(0.6, 0.7, 0.85),
            sun_cascades: SunCascades { matrices, splits },
        }
    }

    fn shadow_pass(
        &mut self,
        sim: &Game,
        _alpha: f32,
        cascade: u32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        // Cascade 0 owns the per-frame walk that fills the instance buckets;
        // cascades 1–3 (and `render`) draw the same buffers, so the walk has
        // to land before any of them. The engine calls `shadow_pass` four
        // times *before* `render`, so cascade 0 is the right anchor.
        if cascade == 0 {
            self.buckets.begin_frame();

            let frustum = Frustum::from_view_proj(self.camera.view_proj());
            RenderObjectTraversal::declare_and_cull(
                &sim.zones,
                &self.templates,
                &mut self.proxies,
                &frustum,
            );

            let asset_server = &self.asset_server;
            let material = &self.material;
            let atlas_material = &self.atlas_material;
            let samplers = &self.samplers;
            let mut adjustments: HashMap<KindId, Mat4> = HashMap::new();
            for (key, template) in &mut self.mesh_templates {
                template
                    .material
                    .refresh(renderer, material, samplers, asset_server);
                adjustments.insert(
                    key.clone(),
                    template.resolve(asset_server).fallback_adjustment,
                );
            }
            for (_, instance) in self.atlas_materials.iter_mut() {
                instance.refresh(renderer, atlas_material, samplers, asset_server);
            }
            // Ground material refreshed alongside the body materials so a
            // late-arriving texture (the checker is `Handle::ready` today,
            // but future debug-toggle paths might force-loading it) flips
            // through the same once-per-frame hook.
            self.ground
                .material
                .refresh(renderer, material, samplers, asset_server);

            // No sim→view per-part state for the editor; the engine already
            // wrote `world_xform`.
            RenderObjectTraversal::update_instances(
                &sim.zones,
                &self.templates,
                &mut self.proxies,
                |_parent, _kind, _components, _instance| {},
            );

            let buckets = &mut self.buckets;
            RenderObjectTraversal::for_each_alive_part(
                &sim.zones,
                &self.templates,
                &self.proxies,
                |_parent, _kind, part, world| {
                    let adjustment = adjustments
                        .get(&part.mesh)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    buckets.push(
                        part.mesh.clone(),
                        MeshInstanceAttribs::new(world * adjustment, Vec4::ONE),
                    );
                },
            );
            self.buckets.upload(&renderer.queue);
        }

        // Depth-only pass against the bucket contents cascade 0 populated.
        // The ground plane is the shadow *receiver* — deliberately not
        // drawn here, so it doesn't shadow itself.
        MeshDraw::depth_only(
            pass,
            renderer,
            &self.asset_server,
            &self.shadow_pipeline,
            &self.mesh_templates,
            &self.buckets,
        );
    }

    fn render(
        &mut self,
        _sim: &Game,
        _alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Bucket population, material refresh, and the cull walk all live
        // in `shadow_pass` (cascade 0) — that runs before `render`, so by
        // now the buckets hold the kind body's instance attrib and the
        // shadow maps are populated for the PBR shader to sample.

        // Camera + scene bind groups (0 + 1) are shared across the PBR /
        // atlas / shadow-receiver pipelines; the per-primitive switch only
        // touches bind group 2 and the pipeline.
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);

        // Ground plane first — opaque shadow receiver under everything else.
        // One instanced draw, identity model matrix.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(2, self.ground.material.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.ground.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.ground.instance_buffer.slice(..));
        pass.set_index_buffer(
            self.ground.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..self.ground.index_count, 0, 0..1);
        renderer.record_draw(1);

        MeshDraw::pbr_with_atlas(
            pass,
            renderer,
            &self.asset_server,
            PbrAtlasMaterials {
                pbr: &self.material,
                atlas: &self.atlas_material,
                atlas_instances: &self.atlas_materials,
            },
            &self.mesh_templates,
            &self.buckets,
        );
    }
}

impl LumberEditorView {
    /// Snap the orbit rig to fit the newly-selected kind's bounds. No-op
    /// when the selection hasn't changed since the previous frame.
    fn maybe_auto_frame(&mut self, sim: &Game) {
        let current = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned();
        if current == self.last_selected {
            return;
        }
        if let Some(kind) = &current
            && let Some(spec) = sim.render_specs.get(kind)
        {
            let aabb = spec.visual_bounds();
            let centre = (aabb.min + aabb.max) * 0.5;
            let extent = (aabb.max - aabb.min).max_element();
            // `extent * 2` keeps the AABB comfortably inside the 45°-ish
            // FOV with margin for surrounding emitter reach; floor at 1 m
            // so a tiny mesh doesn't degenerate to the rig's
            // distance_min clamp.
            self.rig.focus = centre;
            self.rig.distance = (extent * 2.0).max(1.0);
        }
        self.last_selected = current;
    }
}

// --- Ground plane ------------------------------------------------------

/// Edge length of the ground plane in metres. Bigger than any kind the
/// editor is likely to show (lumber-camp's biggest is ~6 m) so the floor
/// always reaches past the visible orbit-rig frustum.
const GROUND_SIZE: f32 = 100.0;
/// World-space size of one checkerboard cell. 25 cm reads as a fine-grained
/// scale reference for sub-metre kinds without becoming visual noise on the
/// larger ones.
const GROUND_CELL_SIZE: f32 = 0.25;

fn build_ground_plane(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
) -> GroundPlane {
    use wgpu::util::DeviceExt;

    // A 2×2 checker baked into a 64×64 texture, tiled across the plane
    // with `LinearRepeat`. UV scale is chosen so each repeat covers two
    // cells (one light + one dark) at `GROUND_CELL_SIZE` metres each.
    let texture = make_checker_texture(renderer);
    let albedo = Handle::ready(texture);
    let ground_material = material.create_instance(
        renderer,
        samplers,
        asset_server,
        PbrMaterialParams {
            albedo,
            sampler: SamplerKind::LinearRepeat,
            albedo_factor: Vec4::ONE,
            // Matte dielectric — the surface should look like rough painted
            // concrete, not a polished display table; keeps the kind's
            // specular highlights the obvious figure-ground signal.
            metallic: 0.0,
            roughness: 0.95,
        },
    );

    // One-quad plane on XY at z=0; UV scaled so `LinearRepeat` tiles the
    // 2×2 checker `GROUND_SIZE / (2 * GROUND_CELL_SIZE)` times per axis.
    let mut mesh = PrimitiveMesh::plane(Vec2::splat(GROUND_SIZE), UVec2::ONE);
    let uv_scale = GROUND_SIZE / (2.0 * GROUND_CELL_SIZE);
    for v in &mut mesh.vertices {
        v.uv[0] *= uv_scale;
        v.uv[1] *= uv_scale;
    }
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    // Static one-instance buffer — identity model, no tint, no hit ID.
    // Never rewritten, so we don't need a separate scratch + upload path.
    let instance = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor ground instance"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX,
        });

    GroundPlane {
        vertex_buffer,
        index_buffer,
        index_count: mesh.index_count(),
        instance_buffer,
        material: ground_material,
    }
}

/// Bake a 64×64 RGBA8 checkerboard intended for `LinearRepeat` tiling. Two
/// soft greys keep the floor reading as a backdrop rather than competing
/// with the displayed kind. Sharp cell edges at 32-px boundaries mean the
/// mip chain handles distant cells cleanly without bleeding the two tones
/// together.
fn make_checker_texture(renderer: &Renderer) -> Texture {
    const SIZE: u32 = 64;
    const CELL_PX: u32 = SIZE / 2;
    const LIGHT: [u8; 4] = [220, 220, 220, 255];
    const DARK: [u8; 4] = [160, 160, 160, 255];
    let mut bytes = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let cx = (x / CELL_PX) & 1;
            let cy = (y / CELL_PX) & 1;
            let c = if cx == cy { LIGHT } else { DARK };
            bytes.extend_from_slice(&c);
        }
    }
    Texture::from_rgba8(renderer, "lumber-editor checker", SIZE, SIZE, &bytes, true)
}

// --- Entry point -------------------------------------------------------

fn main() {
    let vfs = Arc::new(lumber_editor_vfs());
    let kinds_prefix = VfsPath::new("kinds").expect("valid VFS path");
    let defs = pollster::block_on(Definitions::load(&vfs, &kinds_prefix))
        .expect("loading kind definitions");
    let game = Game::new(defs);
    currawong::run::<LumberEditorView>(game);
}
