//! View-side rendering for the lumber camp, on the asset pipeline.
//!
//! Each sim object carries a [`KindId`] component (`currawong:oak_tree`,
//! `currawong:lumberjack`, …) and the view looks up a registered
//! [`RenderTemplate`] keyed by that same id. The mesh + texture for each
//! kind stream through the [`AssetServer`]; the body part of every kind
//! shows the magenta fallback for the first few frames until the
//! background loader thread publishes the real glTF and PNG, then snaps
//! to the real asset on the transition frame.
//!
//! ## Kind → template binding convention (the PR-4 commitment)
//!
//! This is the convention every future example should copy:
//!
//! 1. Sim attaches a [`KindId`] component to each object it wants drawn.
//! 2. View walks [`Definitions`] at init and, for each kind that has a
//!    `render: ( shape: "...", mesh: "...", albedo: "...", ... )` block,
//!    builds one [`RenderTemplate`] keyed by the kind id and registers
//!    it on [`RenderRegistry<KindId, PartKey, PartKey>`].
//! 3. The `render.shape` tag (a small closed set — today
//!    `"tree" | "pawn" | "building"`) selects which view-side factory
//!    runs: the *factory* owns the structural layout (which mesh parts
//!    the template has, their local transforms, their visual AABB), the
//!    *def* feeds it the per-kind data (mesh path, texture path, PBR
//!    factors, bounds). A new kind under an existing shape is a new RON
//!    file. A new shape is a new factory + a new arm in
//!    [`RenderShape::register_template`].
//! 4. The per-instance update dispatches on a precomputed
//!    `HashMap<KindId, RenderShape>` rather than re-matching strings each
//!    frame — keeps the per-frame seam to one HashMap lookup.
//!
//! Sim-vocabulary slot names + view-state-on-`RenderProxy` from
//! [`super`]/CLAUDE.md still apply: the marker and the carried log are
//! view-side visibility decisions on the persistent
//! [`RenderProxy`], driven by the sim's typed [`Designated`] /
//! [`Carrying`] components.

mod debugui;
mod gameui;
mod pawn;
mod tree;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use currawong::data::{Definitions, KindId, VfsPath};
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, FlatTopsMesher, Frustum,
    HitTarget, InstanceBuckets, MaterialId, MaterialRegistry, MeshDraw, MeshInstanceAttribs,
    MeshTemplate, OrbitRig, PbrAtlasMaterial, PbrAtlasMaterialInstance, PbrAtlasMaterialParams,
    PbrAtlasMaterials, PbrMaterial, PbrMaterialInstance, RenderObjectTraversal, RenderProxies,
    RenderRegistry, RenderSpec, RenderTemplate, Renderer, SamplerKind, SamplerRegistry,
    ShadowMeshPipeline, SunCascades, TerrainMaterial, TerrainMaterialInstance, TerrainRenderer,
    TextureColorSpace, View, ViewConfig, ViewEnvironment, WorldObjectRef, YakuiAssets, ZoneId,
    build_hierarchical_render_template, egui, sun_direction_for, wgpu, winit, yakui,
};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use pawn::PawnRenderer;

use crate::sim::{Command, Game, GameState, HEIGHT_UNIT, TILE_SIZE, TIME_LIMIT_SECS, WOOD_GOAL};

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

// --- Part keys ----------------------------------------------------------

/// Names one drawable part in the example's part registry. `Body` is the
/// per-kind shape (mesh + texture streamed from the def's `render` block);
/// `Marker` and `CarriedLog` are shared procedural ancillary parts reused
/// across every tree species and every pawn kind respectively.
///
/// Sharing the ancillary geometry across kinds means a new species
/// reuses one buffer pair rather than allocating its own marker mesh —
/// and the engine's `MeshPart` references a `PartKey` by value, so any
/// kind's template can declare them with no extra registration step.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum PartKey {
    /// The kind's main body — streamed glTF + PNG resolved at init from
    /// the kind's `render` block. Used by the flat schema (kinds whose
    /// `render` block has no `nodes:` field).
    Body(KindId),
    /// Streamed glb referenced by VFS path. Used by the hierarchical
    /// schema (kinds whose `render.nodes` block declares one Mesh node
    /// per glb); deduped across kinds that reference the same path.
    Glb(VfsPath),
    /// Shared red-cone marker shown above designated trees.
    Marker,
    /// Shared brown-cylinder log shown on carrying pawns.
    CarriedLog,
}

impl PartKey {
    /// Body parts receive the hover tint; ancillary parts don't (keeps the
    /// marker readably red, the carried log readably brown, no matter
    /// which body it sits on). Hierarchical Glb nodes are body-like — a
    /// child mesh declared in `render.nodes` is part of the kind's main
    /// visual, not an ancillary attached by the view.
    fn is_body(&self) -> bool {
        matches!(self, Self::Body(_) | Self::Glb(_))
    }
}

// --- Render shape tag --------------------------------------------------

/// View-side discriminator chosen from each kind's `render.shape`. Cached
/// in [`LumberCampView::shapes`] so the per-instance update closure
/// dispatches with one HashMap lookup rather than a string match every
/// frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderShape {
    Tree,
    Pawn,
    Building,
}

impl RenderShape {
    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "tree" => Some(Self::Tree),
            "pawn" => Some(Self::Pawn),
            "building" => Some(Self::Building),
            _ => None,
        }
    }
}

// --- View ---------------------------------------------------------------

/// `RenderRegistry` generics for this view — pinned once to keep the
/// `LumberCampView` field readable.
type Templates = RenderRegistry<KindId, PartKey, PartKey>;

pub struct LumberCampView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,

    material: PbrMaterial,
    /// Stylized atlas-PBR material — used by glb primitives whose
    /// material slot resolves through [`Self::atlas_materials`].
    atlas_material: PbrAtlasMaterial,
    /// Registry of atlas-material instances keyed by glb material name.
    /// Today only `gltf:lumber` is populated (from `assets/lumber/`'s
    /// albedo + MRE atlases); add a new entry per stylized atlas pair.
    atlas_materials: MaterialRegistry<PbrAtlasMaterialInstance>,
    samplers: SamplerRegistry,
    asset_server: AssetServer,
    /// Cache from VFS path → yakui [`ManagedTextureId`](yakui::ManagedTextureId)
    /// for game-UI imagery (the nineslice window-frame PNGs). Shares the same
    /// [`Arc<Vfs>`] as [`asset_server`](Self::asset_server) so mods resolve
    /// identically on the world-asset and UI-asset paths.
    yakui_assets: YakuiAssets,

    /// GPU bundle per drawable part. Looked up by the draw loop after the
    /// per-part bucket has been filled by the engine walk. Refresh runs
    /// each frame inside [`Self::render`] to reconcile the texture handles
    /// (cheap when nothing changed, swaps the bind group on the frame the
    /// real PNG lands).
    mesh_templates: HashMap<PartKey, MeshTemplate<PbrMaterialInstance>>,
    /// Engine-side render-object registry: maps each `KindId` to a
    /// [`RenderTemplate`] declaring which `PartKey`s make up an instance,
    /// where each sits relative to the parent transform, and the visual
    /// AABB the engine frustum-culls against.
    templates: Templates,
    /// Precomputed shape tag per kind. The per-instance update closure
    /// dispatches on this rather than re-matching the def's string tag
    /// every frame.
    shapes: HashMap<KindId, RenderShape>,
    /// Live render-object proxies with cull hysteresis. Owns the
    /// view-side per-instance state (`world_xform`, per-part visibility)
    /// that the per-instance update closure writes each frame.
    proxies: RenderProxies<KindId>,
    /// Per-part instance-attrib buckets. The extract closure pushes one
    /// [`MeshInstanceAttribs`] per visible part; the draw loop iterates
    /// non-empty buckets and issues one indexed-instanced call each.
    buckets: InstanceBuckets<PartKey, MeshInstanceAttribs>,

    terrain: TerrainRenderer,
    terrain_material: TerrainMaterial,
    terrain_solid: TerrainMaterialInstance,

    /// Depth-only pipeline used in the four cascade shadow passes per
    /// frame. Shares the canonical `PosNormalUv` + `MeshInstanceAttribs`
    /// vertex layout that all mesh draws here use, so the same vertex /
    /// instance buffers can be re-bound under the depth-only pipeline.
    shadow_pipeline: ShadowMeshPipeline,

    /// Object currently under the cursor, resolved from the GPU hit-ID
    /// readback (1–3 frame lag is invisible at hover-highlight latency).
    /// Stored as `WorldObjectRef` so it lines up with the parent that
    /// the engine walk receives in callbacks.
    hovered: Option<WorldObjectRef>,

    /// Per-kind view-state for pawns: interpolation snapshots + idle-bob
    /// clock. Read by the per-instance update closure to overwrite
    /// `instance.world_xform`.
    pawn: PawnRenderer,

    /// Wall-clock anchor for the egui FPS overlay. Sampled in `ui()`, so
    /// frametimes reflect the full frame (sim tick + render + UI) regardless
    /// of sim pause state.
    last_frame: Instant,
    /// Rolling frametime samples feeding the top-right FPS chart.
    frame_samples: VecDeque<f32>,
    /// One-line adapter description for the debug-overlay footer. Built at
    /// init from [`Renderer::adapter_info`] so we don't reformat it every
    /// frame.
    adapter_info: String,
}

impl View for LumberCampView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — lumber camp (asset pipeline)",
        clear_colour: wgpu::Color {
            r: 0.45,
            g: 0.65,
            b: 0.95,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
        shadow_map_resolution: Some(2048),
    };

    fn init(renderer: &Renderer) -> Self {
        // Set camera.far once at init — render() used to overwrite it every
        // frame, but `extract_environment` runs *before* `render` and
        // computes cascade matrices from `camera.far`, so the value must be
        // stable from frame zero.
        let camera = Camera {
            far: 200.0,
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(&renderer.device);
        // Park the rig over the centre of the map, tilted down enough to see
        // the whole zone at a comfortable pitch.
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.distance = 22.0;
        rig.pitch = 55.0_f32.to_radians();

        let samplers = SamplerRegistry::new(&renderer.device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());
        let atlas_material = PbrAtlasMaterial::new(renderer, camera_binding.layout());

        // View-side VFS — independent of the one main.rs handed to the
        // sim's `Definitions`. Same on-disk content, separate cache.
        let vfs = Arc::new(crate::lumber_camp_vfs());
        let asset_server = AssetServer::new(renderer, vfs.clone());

        // The glb's "Lumber" material slot resolves through the registry
        // as `gltf:lumber` (the default-namespace mapping the registry
        // applies to bare names from glb). Both atlases stream through
        // the AssetServer like every other texture in this example —
        // magenta-fallback while loading, real view on transition. albedo
        // is sRGB (colour data); mre is linear (R=metallic, G=roughness,
        // B=emission mask).
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
                // Nearest + clamp — low-poly stylization reads the atlas
                // as discrete colour bands, so trilinear blending would
                // muddy the look. Clamp keeps cells from bleeding across
                // edges.
                sampler: SamplerKind::NearestClamp,
            },
        );
        let mut atlas_materials = MaterialRegistry::new();
        atlas_materials.register(
            MaterialId::new("gltf:lumber").expect("valid id"),
            lumber_instance,
        );

        // We also need to re-parse the defs view-side to walk each kind's
        // `render` block. The sim has already validated the file shapes —
        // a failure here would be a build-pipeline divergence, not a
        // runtime condition.
        let defs = currawong::pollster::block_on(Definitions::load(
            &vfs,
            &VfsPath::new("kinds").expect("valid VFS path"),
        ))
        .expect("view-side definitions load");

        // Build the registries by walking the defs.
        let mut mesh_templates: HashMap<PartKey, MeshTemplate<PbrMaterialInstance>> =
            HashMap::new();
        let mut templates: Templates = RenderRegistry::new();
        let mut shapes: HashMap<KindId, RenderShape> = HashMap::new();

        // Shared ancillary parts first — every tree's template references
        // the same `PartKey::Marker`, every pawn's references the same
        // `PartKey::CarriedLog`, so they're one entry each in the
        // mesh-template map.
        mesh_templates.insert(
            PartKey::Marker,
            tree::new_marker_template(renderer, &material, &samplers, &asset_server),
        );
        mesh_templates.insert(
            PartKey::CarriedLog,
            pawn::new_log_template(renderer, &material, &samplers, &asset_server),
        );

        // Walk the defs. For every kind that has a `render` block we
        // recognise, build a body MeshTemplate + a per-kind
        // RenderTemplate. Unknown shapes are logged and skipped — the
        // sim might still place them, in which case the engine cull will
        // simply not find a template and skip them silently.
        for (kind_id, render, body_template) in material.streamed_kind_body_templates(
            renderer,
            &samplers,
            &asset_server,
            &defs,
            |kid, e| eprintln!("lumber_camp: skipping {kid}: {e}"),
        ) {
            let Some(shape) = RenderShape::from_tag(&render.shape) else {
                eprintln!(
                    "lumber_camp: kind {kind_id} declares unknown render.shape `{}`; skipping",
                    render.shape
                );
                continue;
            };
            // Hierarchical schema: walk `render.nodes` and register one
            // glb-keyed MeshTemplate per unique mesh path. The flat
            // `body_template` is unused on this path — its `PartKey::Body`
            // slot is replaced by per-node `PartKey::Glb` slots. Today
            // only the building shape opts into nodes; tree and pawn keep
            // their ancillary-augmented flat templates.
            let template = if !render.nodes.is_empty() {
                build_hierarchical_template(
                    &kind_id,
                    &render,
                    renderer,
                    &material,
                    &samplers,
                    &asset_server,
                    &mut mesh_templates,
                )
            } else {
                mesh_templates.insert(PartKey::Body(kind_id.clone()), body_template);
                shape.register_template(kind_id.clone(), &render)
            };
            templates.register(kind_id.clone(), template);
            shapes.insert(kind_id, shape);
        }

        let proxies = RenderProxies::<KindId>::new(CULL_HYSTERESIS_FRAMES);

        let mut buckets = InstanceBuckets::<PartKey, MeshInstanceAttribs>::new(
            "lumber-camp instances",
            MAX_INSTANCES_PER_PART,
        );
        // Every part key we registered (body keys + the two ancillaries)
        // needs a bucket. The draw loop short-circuits on
        // `mesh_templates.get(...)`, so an absent bucket would silently
        // drop draws — registering up front is the safer shape.
        for key in mesh_templates.keys().cloned().collect::<Vec<_>>() {
            buckets.register(&renderer.device, key);
        }

        let terrain_material = TerrainMaterial::new(renderer, camera_binding.layout());
        let terrain_solid = terrain_material.create_instance(renderer, TERRAIN_TINT);

        let shadow_pipeline = ShadowMeshPipeline::new(renderer);

        let yakui_assets = YakuiAssets::new(vfs.clone());

        Self {
            camera,
            camera_binding,
            rig,
            material,
            atlas_material,
            atlas_materials,
            samplers,
            asset_server,
            yakui_assets,
            mesh_templates,
            templates,
            shapes,
            proxies,
            buckets,
            terrain: TerrainRenderer::new(),
            terrain_material,
            terrain_solid,
            shadow_pipeline,
            hovered: None,
            pawn: PawnRenderer::new(),
            last_frame: Instant::now(),
            frame_samples: VecDeque::with_capacity(debugui::FRAMETIME_HISTORY),
            adapter_info: debugui::format_adapter_info(renderer.adapter_info()),
        }
    }

    fn update(
        &mut self,
        sim: &Game,
        ctx: &mut EngineCtx,
        _: &mut CommandQueue<Command>,
        dt: Duration,
    ) {
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

    fn extract_environment(&self, sim: &Game, _: ZoneId) -> ViewEnvironment {
        // Sun direction tracks `sim.env.time_of_day`; the bottom-right env
        // panel feeds back into that via `Command::SetTimeOfDay`. Colour
        // model (intensity falloff + warm-to-cool tint + sky/ambient dim)
        // mirrors `examples/textured_pbr.rs` so dragging the slider after
        // sunset visibly darkens the scene.
        let sun_direction = sun_direction_for(sim.env.time_of_day);
        let above_horizon = sun_direction.z.max(0.0);
        let intensity = above_horizon * above_horizon * 2.5;
        let warm = Vec3::new(1.0, 0.55, 0.30);
        let cool = Vec3::new(1.0, 0.95, 0.85);
        let warmth = (1.0 - above_horizon).clamp(0.0, 1.0).powf(2.0);
        let sun_color = cool.lerp(warm, warmth) * intensity;

        let day = above_horizon.clamp(0.0, 1.0).powf(0.5);
        let sky_color = Vec3::new(0.45, 0.65, 0.95).lerp(Vec3::new(0.02, 0.03, 0.06), 1.0 - day);
        let ambient = Vec3::new(0.05, 0.06, 0.09).lerp(Vec3::new(0.30, 0.32, 0.38), day);

        // Only fit cascades while the sun is meaningfully above the horizon
        // — at night the lit shader skips shadow sampling via the
        // `splits[0] == 0.0` sentinel.
        let sun_cascades = if above_horizon > 0.05 {
            let splits = self.camera.cascade_split_distances(0.75);
            let matrices = self.camera.fit_shadow_cascades(sun_direction, splits);
            SunCascades { matrices, splits }
        } else {
            SunCascades::disabled()
        };

        ViewEnvironment {
            sun_direction,
            sun_color,
            ambient,
            sky_color,
            sun_cascades,
        }
    }

    fn shadow_pass(
        &mut self,
        sim: &Game,
        alpha: f32,
        cascade: u32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        // First-cascade work: the per-frame walk that populates the
        // instance buckets. The other three cascade passes (and `render`)
        // draw the same buckets, so the walk has to land before any of
        // them — and the engine calls `shadow_pass` four times before
        // `render`, so cascade 0 is the right anchor.
        if cascade == 0 {
            // Resolve the current hover from the GPU readback. `hit_id_hover`
            // can lag rendering by 1–3 frames; for hover-highlight that's
            // invisible. Terrain hits and "no hit" both clear the object
            // hover.
            self.hovered = match renderer.hit_id_hover() {
                Some(HitTarget::Object { zone, id }) => Some(WorldObjectRef { zone, id }),
                _ => None,
            };

            self.buckets.begin_frame();

            // Phase 1: engine-driven walk + declare + cull. One proxy per
            // sim object carrying a `KindId` matching a registered
            // template; frustum-culled with hysteresis.
            let frustum = Frustum::from_view_proj(self.camera.view_proj());
            RenderObjectTraversal::declare_and_cull(
                &sim.zones,
                &self.templates,
                &mut self.proxies,
                &frustum,
            );
            let (visible, hysteresis) = self.proxies.cull_counts();
            renderer.record_proxies(visible, hysteresis);

            // Phase 1.5: reconcile material handles + cache the per-template
            // fallback adjustments. Material `refresh` is cheap when nothing
            // changed; on the frame a streamed texture transitions Loading →
            // Ready the bind group is rebuilt against the real view.
            let adjustments = MeshDraw::refresh_pbr_atlas_materials(
                renderer,
                &self.asset_server,
                &self.samplers,
                &self.material,
                &self.atlas_material,
                &mut self.atlas_materials,
                &mut self.mesh_templates,
            );

            // Phase 1.7: the single sim→view translation seam.
            let pawn = &self.pawn;
            let shapes = &self.shapes;
            RenderObjectTraversal::update_instances(
                &sim.zones,
                &self.templates,
                &mut self.proxies,
                |parent, kind_id, components, template, instance| match shapes.get(kind_id) {
                    Some(RenderShape::Pawn) => {
                        pawn::update_instance(parent, components, template, instance, alpha, pawn)
                    }
                    Some(RenderShape::Tree) => {
                        tree::update_instance(parent, components, template, instance)
                    }
                    Some(RenderShape::Building) | None => {}
                },
            );

            // Phase 2: engine-driven extract — populate per-part buckets
            // with the hit-ID-aware traversal so a click on any part
            // resolves to the same `WorldObjectId`.
            let buckets = &mut self.buckets;
            let hovered = self.hovered;
            RenderObjectTraversal::for_each_alive_part_with_hit_id(
                &sim.zones,
                &self.templates,
                &self.proxies,
                renderer,
                |parent, _kind, part, world, hit_id| {
                    let tint = if hovered == Some(parent) && part.material.is_body() {
                        HOVER_TINT
                    } else {
                        Vec4::ONE
                    };
                    let adjustment = adjustments
                        .get(&part.mesh)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    buckets.push(
                        part.mesh.clone(),
                        MeshInstanceAttribs::new(world * adjustment, tint).with_hit_id(hit_id),
                    );
                },
            );

            self.buckets.upload(&renderer.queue);
        }

        // Same draws every cascade — depth-only pipeline + the bucket
        // contents the cascade-0 walk filled. Terrain is intentionally
        // *not* drawn here: it's the main shadow receiver, not a caster
        // in v1. Hills + cliffs casting would need a depth-only
        // `TerrainVertex` pipeline; deferred.
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
        sim: &Game,
        _alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        // Camera aspect refined per frame from the live surface size; stale-
        // by-one-frame after a resize is acceptable since `extract_environment`
        // (which builds the cascade matrices) reads `self.camera` and the
        // bucket walk above already ran with the same value.
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }
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

        // Terrain first so opaque ground is in the depth buffer before meshes
        // draw on top of it. Same camera + scene env bindings serve both.
        // No liquids in the skeleton — skip the transparent pass.
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.zone, &self.terrain_solid);

        // Mesh parts: one indexed-instanced call per primitive. The atlas
        // dispatch + PBR/atlas pipeline-switching loop is the same shape
        // every kind→template view needs, so the engine helper owns it;
        // we just pass the registries and the buckets. Camera + scene
        // bind groups (0 + 1) carry across both PBR pipelines so the
        // helper only has to touch bind group 2.
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
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

    fn input(
        &mut self,
        _sim: &Game,
        ctx: &mut EngineCtx,
        cmds: &mut CommandQueue<Command>,
        event: &WindowEvent,
    ) {
        self.rig.handle_event(event);
        match event {
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => ctx.event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyF) => self.asset_server.set_force_loading(true),
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Released
                    && matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyF)) =>
            {
                self.asset_server.set_force_loading(false);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => tree::emit_toggle_designation(cmds, self.hovered),
            _ => {}
        }
    }

    fn ui(
        &mut self,
        sim: &Game,
        ctx: &mut EngineCtx,
        cmds: &mut CommandQueue<Command>,
        egui_ctx: &egui::Context,
    ) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_samples.len() == debugui::FRAMETIME_HISTORY {
            self.frame_samples.pop_front();
        }
        self.frame_samples.push_back(dt);
        debugui::show(
            egui_ctx,
            debugui::DebugInput {
                samples: &self.frame_samples,
                timings: &ctx.timings,
                stats: &ctx.stats,
                adapter_info: &self.adapter_info,
            },
        );
        debugui::show_environment(egui_ctx, &sim.env, cmds);
    }

    fn game_ui(
        &mut self,
        sim: &Game,
        ctx: &mut EngineCtx,
        _: &mut CommandQueue<Command>,
        yakui_ctx: &mut yakui::Yakui,
    ) {
        // Cache-or-load the window-frame nineslice. First call blocks on the
        // VFS read + decode + upload to yakui; later calls hit the cache.
        // The PNG is 64×28 with 8-pixel corners; `Pad::all(8.0)` picks out
        // those corners, leaving the stretchy centre for the panel body.
        let frame = self
            .yakui_assets
            .texture(
                yakui_ctx,
                &VfsPath::new("lumber/window_corners.png").expect("valid path"),
            )
            .expect("load lumber/window_corners.png");

        // Top-left status panel: wood progress + countdown. Always present
        // so the player sees the goal even after winning.
        let wood = sim.wood_count();
        // View-side render of a sim-side fixed-point value — convert at
        // the seam.
        let elapsed: f32 = sim.elapsed.to_num();
        let remaining = ((TIME_LIMIT_SECS as f32) - elapsed).max(0.0);
        yakui::pad(yakui::widgets::Pad::all(16.0), || {
            yakui::align(yakui::Alignment::TOP_LEFT, || {
                gameui::panel(
                    frame,
                    yakui::widgets::Pad::all(14.0),
                    yakui::widgets::Pad::all(4.0),
                    yakui::Color::rgba(220, 244, 232, 255),
                    yakui::Color::rgba(20, 24, 32, 220),
                    || {
                        yakui::column(|| {
                            yakui::label(format!("Wood: {wood} / {WOOD_GOAL}"));
                            yakui::label(format!("Time: {}", format_clock(remaining)));
                            if yakui::button("quit").clicked {
                                ctx.event_loop.exit();
                            }
                        });
                    },
                );
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
                gameui::panel(
                    frame,
                    yakui::widgets::Pad::all(14.0),
                    yakui::widgets::Pad::all(4.0),
                    bg,
                    bg,
                    || {
                        yakui::column(|| {
                            yakui::label(text);
                            yakui::label("Esc to quit");
                        });
                    },
                );
            });
        };
    }
}

impl RenderShape {
    /// Build the kind's [`RenderTemplate`] for the given `kind_id` and
    /// `render` spec. Different shapes have different structural layouts —
    /// trees add a marker part, pawns add a carried-log part, buildings
    /// have only a body — so the dispatch happens here rather than at
    /// extract time.
    fn register_template(
        self,
        kind_id: KindId,
        render: &RenderSpec,
    ) -> RenderTemplate<PartKey, PartKey> {
        let body_key = PartKey::Body(kind_id);
        let bounds = render.visual_bounds();
        match self {
            RenderShape::Tree => RenderTemplate::new("tree")
                .with_mesh_part(body_key.clone(), body_key, Mat4::IDENTITY)
                .with_mesh_part(
                    PartKey::Marker,
                    PartKey::Marker,
                    tree::marker_local_transform(&bounds),
                )
                .with_visual_bounds(tree::extended_bounds(&bounds)),
            RenderShape::Pawn => RenderTemplate::new("pawn")
                .with_mesh_part(body_key.clone(), body_key, Mat4::IDENTITY)
                .with_mesh_part(
                    PartKey::CarriedLog,
                    PartKey::CarriedLog,
                    pawn::log_local_transform(&bounds),
                )
                .with_visual_bounds(pawn::extended_bounds(&bounds)),
            RenderShape::Building => RenderTemplate::new("building")
                .with_mesh_part(body_key.clone(), body_key, Mat4::IDENTITY)
                .with_visual_bounds(bounds),
        }
    }
}

/// Format `seconds` as `M:SS` for the HUD countdown.
fn format_clock(seconds: f32) -> String {
    let total = seconds.max(0.0) as u32;
    format!("{}:{:02}", total / 60, total % 60)
}

// --- Hierarchical template build ---------------------------------------

/// Fallback PBR albedo used by Mesh nodes that don't supply their own
/// texture. The atlas pipeline takes over for primitives whose glb
/// material slot resolves through [`MaterialRegistry`] as `gltf:lumber`;
/// this PBR fallback is only bound for non-atlas primitives.
const HIERARCHICAL_FALLBACK_ALBEDO_PATH: &str = "lumber/gradient_atlas.png";

/// Walk `render.nodes` through the shared engine helper, keyed by
/// [`PartKey::Glb`]. Same call shape the lumber_editor uses — both
/// examples dedupe streamed templates by glb path and share PBR
/// material defaults.
fn build_hierarchical_template(
    kind_id: &KindId,
    render: &RenderSpec,
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    mesh_templates: &mut HashMap<PartKey, MeshTemplate<PbrMaterialInstance>>,
) -> RenderTemplate<PartKey, PartKey> {
    let fallback_albedo =
        VfsPath::new(HIERARCHICAL_FALLBACK_ALBEDO_PATH).expect("valid fallback albedo path");
    let label = format!("lumber_camp: kind {kind_id}");
    build_hierarchical_render_template(
        &label,
        render,
        PartKey::Glb,
        &fallback_albedo,
        renderer,
        material,
        samplers,
        asset_server,
        mesh_templates,
    )
}
