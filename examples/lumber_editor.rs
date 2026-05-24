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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{Definitions, FsSource, KindId, Vfs, VfsPath};
use currawong::glam::{Mat4, Quat, UVec2, Vec2, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, Facing, FatLineMaterial,
    FatLineMaterialInstance, FatLineMaterialParams, FatLineVertex, Footprint, Frustum, Handle,
    HandleState, InstanceBuckets, Interaction, MaterialId, MaterialRegistry, MeshBacking, MeshDraw,
    MeshInstanceAttribs, MeshTemplate, OrbitRig, PbrAtlasMaterial, PbrAtlasMaterialInstance,
    PbrAtlasMaterialParams, PbrAtlasMaterials, PbrMaterial, PbrMaterialInstance, PbrMaterialParams,
    PrimitiveMesh, RenderObjectTraversal, RenderProxies, RenderRegistry, RenderSpec,
    RenderTemplate, Renderer, SamplerKind, SamplerRegistry, ShadowMeshPipeline, SimPos, SimUnit,
    Simulation, SunCascades, Texture, TextureColorSpace, View, ViewConfig, ViewEnvironment,
    WorldObjectId, WorldTransform, Zone, ZoneId, Zones, egui, pollster,
    unit_cube_fat_line_geometry, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- VFS ---------------------------------------------------------------

/// Mount the repo's `assets/` directory as a fresh [`Vfs`] with the
/// filesystem watcher running. Called twice at startup — once by `main` for
/// the sim's [`Definitions`], once by the view for the [`AssetServer`] —
/// matching the lumber_camp convention.
///
/// The main-side VFS is dropped immediately after [`Definitions::load`]
/// returns, so its short-lived watcher tears down with it; the cost is one
/// brief notify thread spawn. The view-side VFS is the one that actually
/// drives hot reload through [`AssetServer::pump`].
fn lumber_editor_vfs() -> Vfs {
    let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let mut source = FsSource::new(assets_root);
    if let Err(e) = source.start_watching() {
        eprintln!("lumber_editor: hot reload disabled (watcher init failed: {e})");
    }
    let mut vfs = Vfs::new();
    vfs.mount(source);
    vfs
}

// --- Sim ---------------------------------------------------------------

/// Sim-side mutations: kind selection from the egui panel, plus the
/// hot-reload reload-typed-digest pushed by the View when the file watcher
/// detects a change under `assets/`.
#[derive(Debug, Clone)]
enum Command {
    SelectKind(KindId),
    /// Replace the sim's parsed-at-startup caches with a freshly-loaded
    /// snapshot. The view parses [`Definitions`] off the file-watcher tick
    /// (it's a tiny directory; cheap) and ships the typed digest through
    /// here so [`apply_command`](Game::apply_command) stays a synchronous
    /// in-memory swap.
    ReloadDefinitions {
        available: Vec<KindId>,
        render_specs: HashMap<KindId, RenderSpec>,
        interactions: HashMap<KindId, Interaction>,
        footprints: HashMap<KindId, Footprint>,
    },
    /// Replace the `bounds_min`/`bounds_max` of a kind's cached
    /// [`RenderSpec`] in memory. Drives the recalc-from-mesh button: the
    /// edit lives in sim state and is reflected by the bounds overlay +
    /// camera auto-frame on the next frame. The Save button mirrors the
    /// in-memory value out to the kind's `.ron` file separately.
    UpdateBounds {
        kind: KindId,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    },
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
    /// Cached `Interaction` per kind for the interaction-tiles overlay.
    /// Same shape as `render_specs` — sim-side typed parse of the def, done
    /// once at startup so the view can `.get(&kind).tiles(transform)` each
    /// frame without re-parsing. Kinds whose def omits `interaction:`
    /// deserialize to [`Interaction::None`], which `tiles` resolves to an
    /// empty vec — the overlay simply draws zero instances.
    interactions: HashMap<KindId, Interaction>,
    /// Cached `Footprint` per kind for the placement-tiles overlay. Same
    /// once-at-startup parse pattern as `interactions`; kinds without a
    /// `tiles:` field deserialize to an empty `Footprint` and draw zero
    /// instances.
    footprints: HashMap<KindId, Footprint>,
}

impl Game {
    fn new(defs: Definitions) -> Self {
        let mut available: Vec<KindId> = Vec::new();
        let mut render_specs: HashMap<KindId, RenderSpec> = HashMap::new();
        let mut interactions: HashMap<KindId, Interaction> = HashMap::new();
        let mut footprints: HashMap<KindId, Footprint> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            match RenderSpec::from_def(def) {
                Ok(spec) => {
                    available.push(kind_id.clone());
                    render_specs.insert(kind_id.clone(), spec);
                    // Interaction is independent of render: a kind without a
                    // declared interaction still appears in the editor, just
                    // with an empty tile overlay. A parse error here would
                    // mean a malformed `interaction:` block — eprintln but
                    // fall through to `Interaction::None` so the kind is
                    // still selectable for visual inspection.
                    let interaction = Interaction::from_def(def).unwrap_or_else(|e| {
                        eprintln!("lumber_editor: {kind_id} interaction parse: {e}");
                        Interaction::None
                    });
                    interactions.insert(kind_id.clone(), interaction);
                    // Footprint follows the same opt-in convention; missing
                    // `tiles:` defaults to an empty footprint.
                    let footprint = Footprint::from_def(def).unwrap_or_else(|e| {
                        eprintln!("lumber_editor: {kind_id} footprint parse: {e}");
                        Footprint::default()
                    });
                    footprints.insert(kind_id.clone(), footprint);
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
            interactions,
            footprints,
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
            Command::ReloadDefinitions {
                available,
                render_specs,
                interactions,
                footprints,
            } => {
                self.available = available.clone();
                self.render_specs = render_specs.clone();
                self.interactions = interactions.clone();
                self.footprints = footprints.clone();
                // If the previously-selected kind disappeared from defs
                // (renamed, deleted, or its `render:` block went away),
                // fall back to the first available so the editor doesn't
                // get stuck pointing at a phantom.
                if let Some(zone) = self.zones.get_mut(self.zone) {
                    let current = zone.components().get::<KindId>(self.subject).cloned();
                    let still_valid = current.as_ref().is_some_and(|k| self.available.contains(k));
                    if !still_valid && let Some(first) = self.available.first() {
                        zone.components_mut().insert(self.subject, first.clone());
                    }
                }
            }
            Command::UpdateBounds { kind, min, max } => {
                if let Some(spec) = self.render_specs.get_mut(kind) {
                    spec.bounds_min = *min;
                    spec.bounds_max = *max;
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

    /// Yellow wireframe AABB drawn around the selected kind's visual bounds.
    /// Shares one static unit-cube line buffer across kinds; the per-instance
    /// model matrix is re-uploaded on selection change to scale the unit cube
    /// to the kind's AABB.
    bounds_overlay: BoundsOverlay,

    /// Green flat squares drawn on the ground, one per interaction tile
    /// declared by the selected kind's `Interaction`. Same shape as
    /// `bounds_overlay`: one static unit-quad shared across kinds, one
    /// growable per-frame instance buffer holding a `MeshInstanceAttribs`
    /// per tile. Drawn before the bounds overlay so the yellow AABB lines
    /// read on top of the green squares.
    interaction_overlay: InteractionTilesOverlay,

    /// Orange X-marked squares drawn on the ground, one per placement
    /// tile declared by the selected kind's `Footprint`. Same shape as
    /// `interaction_overlay`, with the unit-square geometry augmented by
    /// two diagonals so a single tile reads as a filled placement marker
    /// rather than just an outline.
    footprint_overlay: FootprintTilesOverlay,

    /// Yellow arrow drawn on the ground from the AABB's front face in
    /// the facing direction, 1 m long. Static shaft + arrowhead vertex
    /// buffer; the per-instance model matrix is rewritten each frame
    /// from the selected kind's AABB and `WorldTransform::facing`.
    facing_arrow_overlay: FacingArrowOverlay,

    /// Previous frame's selection; auto-frame fires when this differs from
    /// the sim's current `KindId` component on `subject`.
    last_selected: Option<KindId>,

    /// View-side visibility toggles driven by the left-panel checkboxes.
    /// Each gates the matching draw block in `render` (and the body's
    /// `depth_only` shadow pass so hidden meshes don't leave a shadow).
    show_main_mesh: bool,
    show_bounding_box: bool,
    show_interaction_tiles: bool,
    show_footprint_tiles: bool,
    show_facing_arrow: bool,

    /// Set in [`Self::update`] when the file watcher reports changes and
    /// definitions re-parse cleanly. Drained in [`Self::shadow_pass`]
    /// (cascade 0) to rebuild `mesh_templates` + `templates`. Two-phase
    /// because `update` has no `Renderer` and the template rebuild needs
    /// one — splitting the parse half (file-system reads) from the GPU
    /// half (handle requests, material instances) keeps the seam clean.
    pending_defs: Option<Definitions>,

    /// Source `.ron` file (relative to the VFS root) each kind was loaded
    /// from. Populated alongside the per-kind template caches in `init` and
    /// `maybe_rebuild_templates`. The Save button uses it to find the file
    /// to rewrite.
    kind_sources: HashMap<KindId, VfsPath>,

    /// On-disk root the VFS is mounted on. Joined with a [`VfsPath`] to get
    /// a real `Path` the editor can write to when the user clicks Save.
    assets_root: PathBuf,

    /// In-memory bounds edit, scoped to one kind at a time. `Some((kind,
    /// pristine))` means the user has clicked Recalc against `kind` and
    /// the sim's `render_specs[kind]` has been updated; `pristine` is the
    /// pre-edit [`RenderSpec`] kept so the edit can be reverted if the
    /// user switches kinds without saving. `None` means no unsaved edit
    /// is pending — the Save button is disabled in that state.
    ///
    /// Switching kinds, clicking Save, or a hot reload all clear this
    /// back to `None` (with a revert command pushed on the switch-without-save
    /// path so sim state matches disk again).
    pending_edit: Option<(KindId, RenderSpec)>,
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

/// GPU resources for the bounding-box overlay. Uses [`FatLineMaterial`] for
/// a constant pixel width regardless of camera distance — wgpu has no
/// `lineWidth` knob, so each segment is expanded to a screen-space quad in
/// the vertex shader. Unit-cube vertex/index buffers are static; the
/// single-instance buffer holds a model matrix that maps `[-0.5, 0.5]³` to
/// the active kind's visual AABB and is rewritten every frame in `render`.
struct BoundsOverlay {
    material: FatLineMaterial,
    color: FatLineMaterialInstance,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
}

/// GPU resources for the interaction-tiles overlay. Four-edge fat-line
/// square in the XY plane shared across kinds (constant 8 px screen-space
/// stroke), with a per-frame instance buffer pre-sized to
/// [`MAX_INTERACTION_TILES`] and rewritten each frame from the selected
/// kind's [`Interaction::tiles`]. Draws zero instances when the selection
/// has [`Interaction::None`].
struct InteractionTilesOverlay {
    material: FatLineMaterial,
    color: FatLineMaterialInstance,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    /// Scratch buffer for assembling per-tile instance attribs before the
    /// single `write_buffer` upload. Kept on the struct so the allocation
    /// is reused across frames.
    instance_scratch: Vec<MeshInstanceAttribs>,
}

/// GPU resources for the placement-tiles overlay. Identical packaging to
/// [`InteractionTilesOverlay`] — fat-line pipeline, per-frame instance
/// buffer pre-sized to [`MAX_FOOTPRINT_TILES`] — but the static
/// vertex/index data is a unit square with two diagonals (six fat-line
/// segments per tile, drawn in orange).
struct FootprintTilesOverlay {
    material: FatLineMaterial,
    color: FatLineMaterialInstance,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_scratch: Vec<MeshInstanceAttribs>,
}

/// GPU resources for the facing-direction arrow overlay. Same fat-line
/// pipeline as the bounds wireframe and tile overlays, but the static
/// vertex/index data is a unit-length shaft along +X plus two
/// arrowhead segments. The per-instance model matrix is rewritten each
/// frame in `render` to translate the shaft origin to the front face
/// of the selected kind's visual AABB and rotate by the subject's
/// `Facing`.
struct FacingArrowOverlay {
    material: FatLineMaterial,
    color: FatLineMaterialInstance,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    instance_buffer: wgpu::Buffer,
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
        let mut kind_sources: HashMap<KindId, VfsPath> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            kind_sources.insert(kind_id.clone(), def.source.clone());
        }

        // Silent `on_skip` — `Game::new` already eprintlned the same errors
        // when it built the sim-side `render_specs` cache.
        for (kind_id, _spec, body) in material.streamed_kind_body_templates(
            renderer,
            &samplers,
            &asset_server,
            &defs,
            |_, _| {},
        ) {
            let bounds = body.visual_bounds;
            mesh_templates.insert(kind_id.clone(), body);
            let template = RenderTemplate::new(kind_id.as_str())
                .with_mesh_part(kind_id.clone(), kind_id.clone(), Mat4::IDENTITY)
                .with_visual_bounds(bounds);
            templates.register(kind_id, template);
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
        let bounds_overlay = build_bounds_overlay(renderer, camera_binding.layout());
        let interaction_overlay = build_interaction_overlay(renderer, camera_binding.layout());
        let footprint_overlay = build_footprint_overlay(renderer, camera_binding.layout());
        let facing_arrow_overlay = build_facing_arrow_overlay(renderer, camera_binding.layout());

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
            bounds_overlay,
            interaction_overlay,
            footprint_overlay,
            facing_arrow_overlay,
            last_selected: None,
            show_main_mesh: true,
            show_bounding_box: true,
            show_interaction_tiles: true,
            show_footprint_tiles: true,
            show_facing_arrow: true,
            pending_defs: None,
            kind_sources,
            assets_root: Path::new(env!("CARGO_MANIFEST_DIR")).join("assets"),
            pending_edit: None,
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
        cmds: &mut CommandQueue<Command>,
        dt: Duration,
    ) {
        // Wall-clock rig integration — keeps WASD pan and scroll zoom
        // responsive regardless of sim speed (the sim doesn't tick anything
        // here anyway, but the invariant still matters).
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);
        self.maybe_auto_frame(sim);
        self.maybe_hot_reload(cmds);
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
                ui.heading("Kind");
                ui.separator();
                let selected_text = current
                    .as_ref()
                    .map(|k| k.as_str().to_string())
                    .unwrap_or_else(|| "Select…".to_string());
                egui::ComboBox::from_id_salt("kind-select")
                    .selected_text(selected_text)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for kind in &sim.available {
                            let selected = current.as_ref() == Some(kind);
                            if ui.selectable_label(selected, kind.as_str()).clicked() && !selected {
                                // Switching away from a kind with an
                                // unsaved bounds edit reverts that edit —
                                // per the editor's "lost if another file
                                // is loaded" model. Done by pushing an
                                // UpdateBounds with the pre-edit pristine
                                // values; the sim applies it on the next
                                // tick, just before SelectKind.
                                if let Some((dirty_kind, pristine)) = self.pending_edit.take() {
                                    cmds.push_now(Command::UpdateBounds {
                                        kind: dirty_kind,
                                        min: pristine.bounds_min,
                                        max: pristine.bounds_max,
                                    });
                                }
                                cmds.push_now(Command::SelectKind(kind.clone()));
                            }
                        }
                    });

                ui.add_space(8.0);
                ui.heading("Visibility");
                ui.separator();
                ui.checkbox(&mut self.show_main_mesh, "Main item mesh");
                ui.checkbox(&mut self.show_bounding_box, "Bounding box");
                ui.checkbox(&mut self.show_interaction_tiles, "Interaction tiles");
                ui.checkbox(&mut self.show_footprint_tiles, "Footprint tiles");
                ui.checkbox(&mut self.show_facing_arrow, "Facing arrow");

                ui.add_space(8.0);
                ui.heading("Bounding box");
                ui.separator();
                // Only enable the button when the mesh is loaded — recalc
                // off the magenta fallback would push its `[-0.5, 0.5]^3`
                // cube into the in-memory spec, which is the wrong
                // direction.
                let mesh_ready = current
                    .as_ref()
                    .and_then(|kind| self.mesh_templates.get(kind))
                    .is_some_and(|tpl| match &tpl.mesh {
                        MeshBacking::Streamed { handle } => handle.is_ready(),
                        MeshBacking::Inline { .. } => true,
                    });
                let recalc_button = egui::Button::new("Recalc bounding box from mesh");
                if ui.add_enabled(mesh_ready, recalc_button).clicked()
                    && let Some(kind) = current.as_ref()
                {
                    self.recalc_bounds_for(kind, sim, cmds);
                }

                // Save button anchored to the bottom of the panel. Bottom-up
                // layout reverses the natural top-down flow, so the button
                // sits flush against the panel's lower edge regardless of
                // how much vertical space the sections above consumed.
                let dirty = current
                    .as_ref()
                    .zip(self.pending_edit.as_ref())
                    .is_some_and(|(k, (dk, _))| k == dk);
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    let save_button = egui::Button::new("Save");
                    if ui
                        .add_enabled(dirty, save_button)
                        .on_hover_text(
                            "Write the in-memory bounds for the selected \
                             kind back to its .ron file.",
                        )
                        .clicked()
                        && let Some(kind) = current.as_ref()
                        && let Some(spec) = sim.render_specs.get(kind)
                    {
                        self.save_bounds_for(kind, spec.visual_bounds());
                    }
                    ui.separator();
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
            // Hot reload's GPU half — if `update` parked a fresh `Definitions`
            // here, rebuild `mesh_templates` + `templates` before anything
            // below dereferences them. The cull/refresh/upload sequence then
            // sees the new handles automatically.
            self.maybe_rebuild_templates(renderer);

            self.buckets.begin_frame();

            let frustum = Frustum::from_view_proj(self.camera.view_proj());
            RenderObjectTraversal::declare_and_cull(
                &sim.zones,
                &self.templates,
                &mut self.proxies,
                &frustum,
            );

            let adjustments = MeshDraw::refresh_pbr_atlas_materials(
                renderer,
                &self.asset_server,
                &self.samplers,
                &self.material,
                &self.atlas_material,
                &mut self.atlas_materials,
                &mut self.mesh_templates,
            );
            // Ground material refreshed alongside the body materials so a
            // late-arriving texture (the checker is `Handle::ready` today,
            // but future debug-toggle paths might force-loading it) flips
            // through the same once-per-frame hook.
            self.ground.material.refresh(
                renderer,
                &self.material,
                &self.samplers,
                &self.asset_server,
            );

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
        // Gated on `show_main_mesh` so hiding the body also hides its
        // shadow — otherwise an invisible kind still casts a silhouette.
        if self.show_main_mesh {
            MeshDraw::depth_only(
                pass,
                renderer,
                &self.asset_server,
                &self.shadow_pipeline,
                &self.mesh_templates,
                &self.buckets,
            );
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

        if self.show_main_mesh {
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

        // Interaction-tiles overlay — green fat-line square outlines on
        // the ground, one per tile returned by the selected kind's
        // `Interaction`. The scratch vec is rebuilt from scratch every
        // frame; cheap for the single-subject editor (a few dozen tiles
        // max in the worst case). Drawn before the bounds wireframe so
        // the yellow AABB lines read on top where the two intersect.
        let zone = sim.zones.get(sim.zone);
        let subject_kind = zone
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .cloned();
        let subject_transform = zone.and_then(|z| z.get(sim.subject)).copied();
        if self.show_interaction_tiles
            && let (Some(kind), Some(transform)) = (subject_kind.as_ref(), subject_transform)
            && let Some(interaction) = sim.interactions.get(kind)
        {
            let tiles = interaction.tiles(&transform);
            let count = self.interaction_overlay.refresh(&renderer.queue, &tiles);
            if count > 0 {
                // FatLine pipeline needs the live viewport size each frame
                // to convert pixel widths into NDC — same write the bounds
                // overlay performs below.
                self.interaction_overlay
                    .color
                    .write_viewport(&renderer.queue, UVec2::new(size.width, size.height));
                pass.set_pipeline(self.interaction_overlay.material.pipeline());
                pass.set_bind_group(1, self.interaction_overlay.color.bind_group(), &[]);
                pass.set_vertex_buffer(0, self.interaction_overlay.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, self.interaction_overlay.instance_buffer.slice(..));
                pass.set_index_buffer(
                    self.interaction_overlay.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                pass.draw_indexed(0..self.interaction_overlay.index_count, 0, 0..count);
                renderer.record_draw(1);
            }
        }

        // Footprint overlay — orange X-marked squares for the selected
        // kind's placement tiles. Same pipeline and shape as the
        // interaction overlay; drawn alongside it so both can be parsed
        // at a glance.
        if self.show_footprint_tiles
            && let (Some(kind), Some(transform)) = (subject_kind.as_ref(), subject_transform)
            && let Some(footprint) = sim.footprints.get(kind)
        {
            let tiles = footprint.tiles(&transform);
            let count = self.footprint_overlay.refresh(&renderer.queue, &tiles);
            if count > 0 {
                self.footprint_overlay
                    .color
                    .write_viewport(&renderer.queue, UVec2::new(size.width, size.height));
                pass.set_pipeline(self.footprint_overlay.material.pipeline());
                pass.set_bind_group(1, self.footprint_overlay.color.bind_group(), &[]);
                pass.set_vertex_buffer(0, self.footprint_overlay.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, self.footprint_overlay.instance_buffer.slice(..));
                pass.set_index_buffer(
                    self.footprint_overlay.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                pass.draw_indexed(0..self.footprint_overlay.index_count, 0, 0..count);
                renderer.record_draw(1);
            }
        }

        // Facing arrow — yellow fat-line arrow on the ground from the
        // AABB's front face in the facing direction. The shaft origin is
        // translated to `position + facing * (aabb.max.x, 0, 0)` and
        // lifted to `FACING_ARROW_Z_EPSILON` so it sits flush with the
        // checker floor without z-fighting. Drawn before the bounding
        // box so the AABB wireframe reads on top where they meet.
        if self.show_facing_arrow
            && let (Some(kind), Some(transform)) = (subject_kind.as_ref(), subject_transform)
            && let Some(spec) = sim.render_specs.get(kind)
        {
            write_facing_arrow_instance(
                &renderer.queue,
                &self.facing_arrow_overlay.instance_buffer,
                transform,
                spec.visual_bounds(),
            );
            self.facing_arrow_overlay
                .color
                .write_viewport(&renderer.queue, UVec2::new(size.width, size.height));
            pass.set_pipeline(self.facing_arrow_overlay.material.pipeline());
            pass.set_bind_group(1, self.facing_arrow_overlay.color.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.facing_arrow_overlay.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.facing_arrow_overlay.instance_buffer.slice(..));
            pass.set_index_buffer(
                self.facing_arrow_overlay.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            pass.draw_indexed(0..self.facing_arrow_overlay.index_count, 0, 0..1);
            renderer.record_draw(1);
        }

        // Bounding-box overlay — yellow fat-line wireframe of the selected
        // kind's visual AABB at a fixed pixel width. Two per-frame uniform
        // writes: (a) the per-instance model matrix from the active
        // selection, (b) the viewport size the vertex shader needs to
        // convert pixels → NDC for the screen-space perpendicular. Drawn
        // last so its depth-tested triangles occlude correctly behind the
        // kind's body but sit on top of co-planar ground geometry.
        let current_aabb = sim
            .zones
            .get(sim.zone)
            .and_then(|z| z.components().get::<KindId>(sim.subject))
            .and_then(|kind| sim.render_specs.get(kind))
            .map(|spec| spec.visual_bounds());
        if self.show_bounding_box
            && let Some(aabb) = current_aabb
        {
            write_bounds_instance(&renderer.queue, &self.bounds_overlay.instance_buffer, aabb);
            self.bounds_overlay
                .color
                .write_viewport(&renderer.queue, UVec2::new(size.width, size.height));
            pass.set_pipeline(self.bounds_overlay.material.pipeline());
            pass.set_bind_group(1, self.bounds_overlay.color.bind_group(), &[]);
            pass.set_vertex_buffer(0, self.bounds_overlay.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.bounds_overlay.instance_buffer.slice(..));
            pass.set_index_buffer(
                self.bounds_overlay.index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );
            pass.draw_indexed(0..self.bounds_overlay.index_count, 0, 0..1);
            renderer.record_draw(1);
        }
    }
}

impl LumberEditorView {
    /// Drain VFS-side change events from the [`AssetServer`]; if anything
    /// changed, re-parse [`Definitions`], ship the typed digest to the sim
    /// via [`Command::ReloadDefinitions`], and park the parsed registry in
    /// [`Self::pending_defs`] for `shadow_pass` to consume.
    ///
    /// Catches three kinds of edit in one flow:
    /// - **Asset bytes** (glb / png) — `pump` evicts the cache entry; the
    ///   template rebuild in `shadow_pass` re-requests fresh handles.
    /// - **`.ron` def fields** (bounds, metallic, footprint…) — re-parsed
    ///   defs feed both the sim's typed caches and the next template
    ///   rebuild.
    /// - **`.ron` adds/removes** (new kind file, deleted kind file) — the
    ///   typed digest's `available` list changes; sim swaps it and resets
    ///   the subject if its KindId was deleted.
    ///
    /// A parse error during reload is logged and dropped — the editor
    /// keeps its previous defs rather than crashing on a half-saved file.
    fn maybe_hot_reload(&mut self, cmds: &mut CommandQueue<Command>) {
        let changed = self.asset_server.pump();
        if changed.is_empty() {
            return;
        }
        // A hot reload replaces sim.render_specs wholesale from disk, so
        // any in-memory bounds edit (this kind's or another's) is gone
        // regardless. Clear the dirty marker to match.
        self.pending_edit = None;
        let defs = match pollster::block_on(Definitions::load(
            self.asset_server.vfs(),
            &VfsPath::new("kinds").expect("valid VFS path"),
        )) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("lumber_editor: hot reload — definitions re-parse failed: {e}");
                return;
            }
        };
        let mut available: Vec<KindId> = Vec::new();
        let mut render_specs: HashMap<KindId, RenderSpec> = HashMap::new();
        let mut interactions: HashMap<KindId, Interaction> = HashMap::new();
        let mut footprints: HashMap<KindId, Footprint> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            let Ok(spec) = RenderSpec::from_def(def) else {
                continue;
            };
            available.push(kind_id.clone());
            render_specs.insert(kind_id.clone(), spec);
            interactions.insert(
                kind_id.clone(),
                Interaction::from_def(def).unwrap_or(Interaction::None),
            );
            footprints.insert(
                kind_id.clone(),
                Footprint::from_def(def).unwrap_or_default(),
            );
        }
        available.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        eprintln!(
            "lumber_editor: hot reload — {} change(s), {} kind(s) after re-parse",
            changed.len(),
            available.len()
        );
        cmds.push_now(Command::ReloadDefinitions {
            available,
            render_specs,
            interactions,
            footprints,
        });
        self.pending_defs = Some(defs);
    }

    /// Rebuild `mesh_templates` + `templates` (and register any new bucket
    /// keys) from `self.pending_defs`. No-op when no reload is pending.
    /// Runs once per reload, at the top of `shadow_pass` cascade 0 where
    /// the `Renderer` is available.
    ///
    /// Templates for kinds removed in the new defs aren't explicitly
    /// dropped from `buckets` (no API for that today) — they simply stop
    /// getting `push`ed because nothing in the cull walk references them.
    /// The bucket buffer lingers as a small leak; fine for editor scope.
    fn maybe_rebuild_templates(&mut self, renderer: &Renderer) {
        let Some(defs) = self.pending_defs.take() else {
            return;
        };
        let mut mesh_templates: HashMap<KindId, MeshTemplate<PbrMaterialInstance>> = HashMap::new();
        let mut templates: Templates = RenderRegistry::new();
        let mut kind_sources: HashMap<KindId, VfsPath> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            kind_sources.insert(kind_id.clone(), def.source.clone());
        }
        for (kind_id, _spec, body) in self.material.streamed_kind_body_templates(
            renderer,
            &self.samplers,
            &self.asset_server,
            &defs,
            |_, _| {},
        ) {
            let bounds = body.visual_bounds;
            mesh_templates.insert(kind_id.clone(), body);
            let template = RenderTemplate::new(kind_id.as_str())
                .with_mesh_part(kind_id.clone(), kind_id.clone(), Mat4::IDENTITY)
                .with_visual_bounds(bounds);
            templates.register(kind_id, template);
        }
        for key in mesh_templates.keys().cloned().collect::<Vec<_>>() {
            self.buckets.register(&renderer.device, key);
        }
        self.mesh_templates = mesh_templates;
        self.templates = templates;
        self.kind_sources = kind_sources;
        // Invalidate the auto-frame cache so a kind whose visual bounds
        // changed reframes the camera on the next `update` tick.
        self.last_selected = None;
    }

    /// Recalculate the visual AABB for `kind` from its loaded mesh and
    /// push an [`Command::UpdateBounds`] so the sim's `render_specs[kind]`
    /// reflects the new value on the next tick. The edit stays in memory
    /// — the Save button is what mirrors it out to disk.
    ///
    /// Snapshots the pre-edit [`RenderSpec`] into `pending_edit` on the
    /// first recalc for a given kind so the edit can be reverted if the
    /// user switches kinds without saving. Subsequent recalcs on the same
    /// kind reuse the same pristine snapshot — clicking recalc N times
    /// in a row is still "one edit" from the dirty-tracking perspective.
    ///
    /// Quietly no-ops if the mesh isn't `Ready` (button is gated in `ui`
    /// too, but state can change between the check and the click).
    fn recalc_bounds_for(&mut self, kind: &KindId, sim: &Game, cmds: &mut CommandQueue<Command>) {
        let Some(template) = self.mesh_templates.get(kind) else {
            eprintln!("lumber_editor: recalc — no mesh template for {kind}");
            return;
        };
        let bounds = match &template.mesh {
            MeshBacking::Streamed { handle } => match handle.peek() {
                HandleState::Ready(mesh) => mesh.bounds,
                HandleState::Loading => {
                    eprintln!("lumber_editor: recalc — mesh for {kind} is still loading");
                    return;
                }
                HandleState::Failed(err) => {
                    eprintln!("lumber_editor: recalc — mesh for {kind} failed: {err}");
                    return;
                }
            },
            // Inline templates don't carry a runtime `Mesh.bounds`; not
            // produced by the editor today, but skip cleanly if one ever
            // appears rather than silently writing a degenerate AABB.
            MeshBacking::Inline { .. } => {
                eprintln!("lumber_editor: recalc — {kind} uses an inline mesh, no recalc");
                return;
            }
        };
        if self.pending_edit.as_ref().is_none_or(|(k, _)| k != kind)
            && let Some(spec) = sim.render_specs.get(kind)
        {
            self.pending_edit = Some((kind.clone(), spec.clone()));
        }
        cmds.push_now(Command::UpdateBounds {
            kind: kind.clone(),
            min: (bounds.min.x, bounds.min.y, bounds.min.z),
            max: (bounds.max.x, bounds.max.y, bounds.max.z),
        });
        // The new bounds may differ enough from the disk-loaded ones that
        // the camera should refit. Invalidate the auto-frame cache so
        // `maybe_auto_frame` runs again on the next `update`.
        self.last_selected = None;
    }

    /// Mirror the sim's current in-memory bounds for `kind` out to its
    /// source `.ron` file and clear `pending_edit`. The file watcher
    /// picks up the write and triggers a hot reload through
    /// [`Self::maybe_hot_reload`]; since disk now matches in-memory the
    /// resulting `ReloadDefinitions` is a no-op for the bounds.
    fn save_bounds_for(&mut self, kind: &KindId, bounds: Aabb) {
        let Some(source) = self.kind_sources.get(kind) else {
            eprintln!("lumber_editor: save — no source path known for {kind}");
            return;
        };
        let on_disk = self.assets_root.join(source.as_str());
        if let Err(e) = rewrite_bounds_in_ron(&on_disk, bounds) {
            eprintln!(
                "lumber_editor: save — failed to rewrite {}: {e}",
                on_disk.display()
            );
            return;
        }
        eprintln!(
            "lumber_editor: save — wrote bounds_min={:?}, bounds_max={:?} to {}",
            bounds.min,
            bounds.max,
            on_disk.display()
        );
        self.pending_edit = None;
    }

    /// Snap the orbit rig to fit the newly-selected kind's bounds. No-op
    /// when the selection hasn't changed since the previous frame.
    /// (Bounds overlay re-upload lives in `render` where the queue is
    /// available — `EngineCtx` deliberately doesn't expose `Renderer`.)
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

// --- Bounds overlay ----------------------------------------------------

/// Yellow used for the bounding-box wireframe. Saturated primary so the
/// box reads against both the lit kind body and the muted ground checker.
const BOUNDS_COLOR: Vec4 = Vec4::new(1.0, 1.0, 0.0, 1.0);
/// Screen-space line width in pixels. Thick enough to read clearly at the
/// editor's typical orbit distance without dominating the figure.
const BOUNDS_WIDTH_PX: f32 = 2.5;

fn build_bounds_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> BoundsOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: BOUNDS_COLOR,
            width_px: BOUNDS_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_cube_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    // Placeholder instance — rewritten every frame in `render` from the
    // active kind's AABB. Pre-sized to MeshInstanceAttribs so the buffer
    // never has to grow.
    let placeholder = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor bounds instance"),
            contents: bytemuck::bytes_of(&placeholder),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    BoundsOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
    }
}

/// Rewrite the bounds-overlay instance buffer so the unit cube maps to
/// `aabb`. Scales `[-0.5, 0.5]³` by `(max - min)` and translates to the
/// AABB centre.
fn write_bounds_instance(queue: &wgpu::Queue, buffer: &wgpu::Buffer, aabb: Aabb) {
    let scale = aabb.max - aabb.min;
    let model = Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, aabb.center());
    let attribs = MeshInstanceAttribs::new(model, Vec4::ONE);
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&attribs));
}

// --- Interaction-tiles overlay ----------------------------------------

/// Saturated lime green for the interaction-tile outlines. Reads cleanly
/// against the muted ground checker and stays clear of the yellow bounds
/// wireframe so the two overlays can be parsed at a glance.
const INTERACTION_TILE_COLOR: Vec4 = Vec4::new(0.20, 0.95, 0.35, 1.0);
/// Screen-space stroke width for the outline in pixels. Picked to match
/// the user-visible weight requested in the editor — heavy enough to read
/// as a deliberate marker, light enough not to dominate small kinds.
const INTERACTION_TILE_WIDTH_PX: f32 = 8.0;
/// Vertical lift above the ground plane to avoid z-fighting with the
/// checker. Small enough to be visually flush, large enough that depth
/// quantization at the far end of the orbit-rig view never collapses it
/// back to the ground.
const INTERACTION_TILE_Z_EPSILON: f32 = 0.01;
/// Cap on the per-frame instance count. `Surround { radius_tiles: 5 }`
/// yields 120 tiles; 256 leaves comfortable headroom for the editor's
/// single-subject scope without ever growing the buffer mid-frame.
const MAX_INTERACTION_TILES: u32 = 256;

/// Fat-line vertex + index data for a unit square outline spanning
/// `[-0.5, 0.5]² × {0}` in XY. Four edges → 16 verts, 24 triangle-list
/// indices. Sibling to [`unit_cube_fat_line_geometry`] — same packing
/// convention so the same [`FatLineMaterial`] pipeline draws both.
fn unit_square_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let h = 0.5;
    let corners: [[f32; 3]; 4] = [
        [-h, -h, 0.0], // 0: -X -Y
        [h, -h, 0.0],  // 1: +X -Y
        [h, h, 0.0],   // 2: +X +Y
        [-h, h, 0.0],  // 3: -X +Y
    ];
    // CCW when viewed from +Z (the orbit-rig camera looks down at the
    // ground), matching the pipeline's `FrontFace::Ccw`.
    let edges: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

    let mut vertices = Vec::with_capacity(16);
    let mut indices = Vec::with_capacity(24);
    for (a, b) in edges {
        let pos_a = corners[a];
        let pos_b = corners[b];
        let base = vertices.len() as u16;
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

fn build_interaction_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> InteractionTilesOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: INTERACTION_TILE_COLOR,
            width_px: INTERACTION_TILE_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_square_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor interaction-tile vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor interaction-tile indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lumber-editor interaction-tile instances"),
        size: (MAX_INTERACTION_TILES as u64) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    InteractionTilesOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
        instance_scratch: Vec::with_capacity(MAX_INTERACTION_TILES as usize),
    }
}

// --- Footprint-tiles overlay ------------------------------------------

/// Saturated orange for the placement tile markers. Sits clearly between
/// the green interaction tiles and the yellow bounds wireframe so all
/// three overlays can be enabled simultaneously and still parsed at a
/// glance.
const FOOTPRINT_TILE_COLOR: Vec4 = Vec4::new(1.0, 0.55, 0.10, 1.0);
/// Screen-space stroke width for the outline + diagonals in pixels.
/// Matches the interaction-tile weight so neighbouring squares read as
/// the same class of marker.
const FOOTPRINT_TILE_WIDTH_PX: f32 = 8.0;
/// Vertical lift above the ground plane to avoid z-fighting. Slightly
/// above the interaction-tile epsilon so the two overlays don't fight
/// each other when they happen to land on the same cell.
const FOOTPRINT_TILE_Z_EPSILON: f32 = 0.015;
/// Cap on the per-frame instance count. A footprint of "thousands of
/// tiles" doesn't make sense; 256 is generous for any plausibly-authored
/// kind and matches the interaction overlay's cap.
const MAX_FOOTPRINT_TILES: u32 = 256;

/// Fat-line vertex + index data for a unit square *with two diagonals*
/// spanning `[-0.5, 0.5]² × {0}` in XY. Six segments (4 edges + 2
/// diagonals) → 24 verts, 36 triangle-list indices. Sibling to
/// [`unit_square_fat_line_geometry`] above; the diagonals are what
/// visually distinguishes a placement tile from a (plain-outline)
/// interaction tile.
fn unit_square_with_diagonals_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let h = 0.5;
    let corners: [[f32; 3]; 4] = [
        [-h, -h, 0.0], // 0: -X -Y
        [h, -h, 0.0],  // 1: +X -Y
        [h, h, 0.0],   // 2: +X +Y
        [-h, h, 0.0],  // 3: -X +Y
    ];
    // 4 perimeter edges + 2 diagonals forming an X across the square.
    // CCW perimeter orientation matches the pipeline's `FrontFace::Ccw`.
    let segments: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (a, b) in segments {
        let pos_a = corners[a];
        let pos_b = corners[b];
        let base = vertices.len() as u16;
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

fn build_footprint_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> FootprintTilesOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: FOOTPRINT_TILE_COLOR,
            width_px: FOOTPRINT_TILE_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_square_with_diagonals_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor footprint-tile vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor footprint-tile indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lumber-editor footprint-tile instances"),
        size: (MAX_FOOTPRINT_TILES as u64) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    FootprintTilesOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
        instance_scratch: Vec::with_capacity(MAX_FOOTPRINT_TILES as usize),
    }
}

impl FootprintTilesOverlay {
    /// Rebuild the per-instance buffer from `tiles` (integer world tile
    /// coords from [`Footprint::tiles`]). Same centring convention as
    /// [`InteractionTilesOverlay::refresh`] — each square is centred on
    /// the integer world coord, matching how the editor draws the kind
    /// body and its interaction tiles. Clamped to [`MAX_FOOTPRINT_TILES`].
    fn refresh(&mut self, queue: &wgpu::Queue, tiles: &[(i32, i32, i32)]) -> u32 {
        self.instance_scratch.clear();
        let cap = MAX_FOOTPRINT_TILES as usize;
        for &(tx, ty, tz) in tiles.iter().take(cap) {
            let translation = Vec3::new(tx as f32, ty as f32, tz as f32 + FOOTPRINT_TILE_Z_EPSILON);
            let model =
                Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, translation);
            self.instance_scratch
                .push(MeshInstanceAttribs::new(model, Vec4::ONE));
        }
        let count = self.instance_scratch.len() as u32;
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }
        count
    }
}

impl InteractionTilesOverlay {
    /// Rebuild the per-instance buffer from `tiles` (integer world tile
    /// coords from [`Interaction::tiles`]). Returns the instance count to
    /// draw — clamped to [`MAX_INTERACTION_TILES`] so a future radius-10
    /// surround can't overflow the buffer (it would just clip silently;
    /// fine for an editor preview, the alternative is a panic mid-render).
    ///
    /// Each square is centred on the *integer* world coord `(tx, ty)` — not
    /// `(tx+0.5, ty+0.5)`. The kind body is drawn at
    /// `transform.position.to_vec3()` (which is the tile-corner value), so
    /// the editor treats the body as "occupying the cell centred on its
    /// origin", and the surrounding interaction squares ring it cleanly at
    /// `(±1, 0)`, `(0, ±1)`, etc. The `+0.5` tile-corner→tile-centre offset
    /// is therefore the wrong convention for this visualisation.
    fn refresh(&mut self, queue: &wgpu::Queue, tiles: &[(i32, i32, i32)]) -> u32 {
        self.instance_scratch.clear();
        let cap = MAX_INTERACTION_TILES as usize;
        for &(tx, ty, tz) in tiles.iter().take(cap) {
            let translation =
                Vec3::new(tx as f32, ty as f32, tz as f32 + INTERACTION_TILE_Z_EPSILON);
            let model =
                Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, translation);
            self.instance_scratch
                .push(MeshInstanceAttribs::new(model, Vec4::ONE));
        }
        let count = self.instance_scratch.len() as u32;
        if count > 0 {
            queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }
        count
    }
}

// --- Facing-arrow overlay --------------------------------------------

/// Saturated yellow for the facing-direction arrow. Matches the bounds
/// wireframe — both describe the selected kind's orientation envelope.
const FACING_ARROW_COLOR: Vec4 = Vec4::new(1.0, 1.0, 0.0, 1.0);
/// Screen-space stroke width for the arrow in pixels. Thick enough to
/// read clearly against the checker floor; heavier than the bounds
/// wireframe so the two overlays don't visually conflate where they
/// share the yellow tone.
const FACING_ARROW_WIDTH_PX: f32 = 6.0;
/// Length of the arrow shaft in metres, measured outward from the AABB
/// front face in the facing direction. Per the user-facing spec for the
/// editor.
const FACING_ARROW_LENGTH: f32 = 1.0;
/// Length of each arrowhead segment as a fraction of the shaft length.
/// Picked to read clearly at typical orbit distances without dominating
/// the shaft.
const FACING_ARROW_HEAD_FRAC: f32 = 0.25;
/// Vertical lift above the ground plane to avoid z-fighting with the
/// checker. Slightly above the footprint tiles' epsilon so the arrow
/// reads on top where they overlap.
const FACING_ARROW_Z_EPSILON: f32 = 0.02;

/// Fat-line vertex + index data for a unit-length facing arrow in local
/// space: shaft from `(0, 0, 0)` to `(FACING_ARROW_LENGTH, 0, 0)`, plus
/// two arrowhead segments forming the tip. Three segments → 12 verts,
/// 18 triangle-list indices. Sibling to [`unit_square_fat_line_geometry`].
fn unit_facing_arrow_fat_line_geometry() -> (Vec<FatLineVertex>, Vec<u16>) {
    let length = FACING_ARROW_LENGTH;
    let head = length * FACING_ARROW_HEAD_FRAC;
    let tip: [f32; 3] = [length, 0.0, 0.0];
    let base_x = length - head;
    // Shaft + two arrowhead lines. All Z = 0; lifted off the ground via
    // the model matrix in `write_facing_arrow_instance`.
    let segments: [([f32; 3], [f32; 3]); 3] = [
        ([0.0, 0.0, 0.0], tip),
        (tip, [base_x, head, 0.0]),
        (tip, [base_x, -head, 0.0]),
    ];

    let mut vertices = Vec::with_capacity(12);
    let mut indices = Vec::with_capacity(18);
    for (pos_a, pos_b) in segments {
        let base = vertices.len() as u16;
        for &(endpoint, side) in &[(0.0_f32, -1.0_f32), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
            vertices.push(FatLineVertex {
                pos_a,
                pos_b,
                side,
                endpoint,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    (vertices, indices)
}

fn build_facing_arrow_overlay(
    renderer: &Renderer,
    camera_layout: &wgpu::BindGroupLayout,
) -> FacingArrowOverlay {
    use wgpu::util::DeviceExt;

    let material = FatLineMaterial::new(renderer, camera_layout);
    let color = material.create_instance(
        renderer,
        FatLineMaterialParams {
            base_color: FACING_ARROW_COLOR,
            width_px: FACING_ARROW_WIDTH_PX,
        },
    );

    let (vertices, indices) = unit_facing_arrow_fat_line_geometry();
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let placeholder = MeshInstanceAttribs::new(Mat4::IDENTITY, Vec4::ONE);
    let instance_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lumber-editor facing-arrow instance"),
            contents: bytemuck::bytes_of(&placeholder),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    FacingArrowOverlay {
        material,
        color,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        instance_buffer,
    }
}

/// Rewrite the facing-arrow instance buffer so the unit-length local-X
/// shaft starts at the AABB's front face (in the subject's facing
/// direction) and points outward along that direction. `Facing` is
/// yaw-only, so the rotation only affects X/Y — the world-space Z stays
/// at [`FACING_ARROW_Z_EPSILON`] regardless.
fn write_facing_arrow_instance(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    transform: WorldTransform,
    aabb: Aabb,
) {
    let position = transform.position.to_vec3();
    let rotation = transform.facing.to_quat();
    let aabb_offset = rotation * Vec3::new(aabb.max.x, 0.0, 0.0);
    let translation = Vec3::new(
        position.x + aabb_offset.x,
        position.y + aabb_offset.y,
        FACING_ARROW_Z_EPSILON,
    );
    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, translation);
    let attribs = MeshInstanceAttribs::new(model, Vec4::ONE);
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&attribs));
}

/// Bake a 64×64 RGBA8 checkerboard intended for `LinearRepeat` tiling. Two
/// soft greys keep the floor reading as a backdrop rather than competing
/// with the displayed kind. Sharp cell edges at 32-px boundaries mean the
/// mip chain handles distant cells cleanly without bleeding the two tones
/// together.
fn make_checker_texture(renderer: &Renderer) -> Texture {
    const SIZE: u32 = 64;
    const CELL_PX: u32 = SIZE / 2;
    const LIGHT: [u8; 4] = [160, 160, 160, 255];
    const DARK: [u8; 4] = [110, 110, 110, 255];
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

// --- Bounds rewrite ----------------------------------------------------

/// Rewrite the `bounds_min:` and `bounds_max:` lines of a kind `.ron` file
/// in place from `bounds`. Walks the file line by line so comments,
/// formatting, and unrelated fields are preserved — the only edits are
/// the two tuple values inside the `render:` block. Whitespace at the
/// start of each line is kept so the indentation matches the source.
///
/// A degenerate match (a kind file with two `bounds_min:` lines, or one
/// inside a string literal) would rewrite both. Acceptable for the
/// editor's authored content; not robust against adversarial RON.
fn rewrite_bounds_in_ron(path: &Path, bounds: Aabb) -> std::io::Result<()> {
    let original = std::fs::read_to_string(path)?;
    let mut out = String::with_capacity(original.len());
    for line in original.split_inclusive('\n') {
        if let Some(rewritten) = rewrite_bounds_line(line, bounds) {
            out.push_str(&rewritten);
        } else {
            out.push_str(line);
        }
    }
    std::fs::write(path, out)
}

/// Return a replacement for `line` if it's a `bounds_min:` / `bounds_max:`
/// line; otherwise `None`. Preserves leading indentation and the original
/// line ending. `Debug` formatting on the f32 components matches the
/// existing RON style (`0.0` not `0`, `-0.5` not `-0.50000`).
fn rewrite_bounds_line(line: &str, bounds: Aabb) -> Option<String> {
    let stripped = line.strip_suffix("\r\n").or(line.strip_suffix('\n'));
    let body = stripped.unwrap_or(line);
    let line_end = &line[body.len()..];
    let trimmed = body.trim_start();
    let indent = &body[..body.len() - trimmed.len()];
    let (field, v) = if trimmed.starts_with("bounds_min:") {
        ("bounds_min", bounds.min)
    } else if trimmed.starts_with("bounds_max:") {
        ("bounds_max", bounds.max)
    } else {
        return None;
    };
    Some(format!(
        "{indent}{field}: ({:?}, {:?}, {:?}),{line_end}",
        v.x, v.y, v.z,
    ))
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
