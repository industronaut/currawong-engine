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
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, Facing, Frustum, InstanceBuckets,
    MaterialId, MaterialRegistry, MeshInstanceAttribs, MeshTemplate, OrbitRig, PbrAtlasMaterial,
    PbrAtlasMaterialInstance, PbrAtlasMaterialParams, PbrMaterial, PbrMaterialInstance,
    RenderObjectTraversal, RenderProxies, RenderRegistry, RenderSpec, RenderTemplate, Renderer,
    SamplerKind, SamplerRegistry, SimPos, SimUnit, Simulation, SunCascades, TextureColorSpace,
    View, ViewConfig, ViewEnvironment, WorldObjectId, WorldTransform, Zone, ZoneId, Zones, egui,
    pollster, wgpu, winit,
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

/// Whether the PBR-or-atlas pipeline is currently bound, tracked across the
/// draw loop so we only flip on transitions (same pattern as lumber_camp).
#[derive(PartialEq, Eq)]
enum ActivePipeline {
    None,
    Pbr,
    Atlas,
}

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

    /// Previous frame's selection; auto-frame fires when this differs from
    /// the sim's current `KindId` component on `subject`.
    last_selected: Option<KindId>,
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
        shadow_map_resolution: None,
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
        // upper-right, modest ambient, neutral sky. `SunCascades::disabled`
        // pairs with `shadow_map_resolution: None` to skip the shadow phase.
        ViewEnvironment {
            sun_direction: Vec3::new(0.4, 0.3, 0.8).normalize(),
            sun_color: Vec3::splat(2.5),
            ambient: Vec3::new(0.25, 0.27, 0.32),
            sky_color: Vec3::new(0.6, 0.7, 0.85),
            sun_cascades: SunCascades::disabled(),
        }
    }

    fn render(
        &mut self,
        sim: &Game,
        _alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }
        self.camera_binding.write(&renderer.queue, &self.camera);

        self.buckets.begin_frame();

        // Phase 1: declare + cull. One proxy for the single sim object
        // carrying a `KindId` matching a registered template.
        let frustum = Frustum::from_view_proj(self.camera.view_proj());
        RenderObjectTraversal::declare_and_cull(
            &sim.zones,
            &self.templates,
            &mut self.proxies,
            &frustum,
        );

        // Phase 1.5: refresh streamed material handles. Cheap when nothing
        // changed; rebuilds the bind group on the frame an asset finishes
        // loading.
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

        // Phase 1.7: per-instance update. No sim → view state translation
        // needed here — the engine already wrote `world_xform` from the
        // sim object's transform, and the editor has no per-part
        // visibility toggling. Still invoked so the engine's state writes
        // land.
        RenderObjectTraversal::update_instances(
            &sim.zones,
            &self.templates,
            &mut self.proxies,
            |_parent, _kind, _components, _instance| {},
        );

        // Phase 2: extract — push one instance attrib per alive part.
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

        // Draw. Camera + scene bind groups (0 + 1) are shared across both
        // PBR pipelines; per-primitive switch only touches bind group 2
        // and the pipeline itself when the glb names an atlas slot.
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        let mut active = ActivePipeline::None;
        for (part_key, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(template) = self.mesh_templates.get(part_key) else {
                continue;
            };
            let resolved = template.resolve(&self.asset_server);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            for prim in resolved.primitives {
                let atlas = prim
                    .material_name
                    .as_deref()
                    .and_then(|name| self.atlas_materials.get_by_name(name));
                match atlas {
                    Some(instance) => {
                        if active != ActivePipeline::Atlas {
                            pass.set_pipeline(self.atlas_material.pipeline());
                            active = ActivePipeline::Atlas;
                        }
                        pass.set_bind_group(2, instance.bind_group(), &[]);
                    }
                    None => {
                        if active != ActivePipeline::Pbr {
                            pass.set_pipeline(self.material.pipeline());
                            active = ActivePipeline::Pbr;
                        }
                        pass.set_bind_group(2, template.material.bind_group(), &[]);
                    }
                }
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..count);
                renderer.record_draw(count);
            }
        }
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

// --- Entry point -------------------------------------------------------

fn main() {
    let vfs = Arc::new(lumber_editor_vfs());
    let kinds_prefix = VfsPath::new("kinds").expect("valid VFS path");
    let defs = pollster::block_on(Definitions::load(&vfs, &kinds_prefix))
        .expect("loading kind definitions");
    let game = Game::new(defs);
    currawong::run::<LumberEditorView>(game);
}
