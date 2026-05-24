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
//! - [`sim::Game`] owns one zone with one [`WorldTransform`] at the origin.
//!   Its sole sim mutation is `SelectKind(KindId)`, which swaps the
//!   [`KindId`] component on the single object.
//! - [`LumberEditorView`] mirrors lumber_camp's kind → template pattern: walk
//!   [`Definitions`] at init, build one [`RenderTemplate`] per kind that has
//!   a `render` block, and dispatch by `KindId` at draw time.
//! - The view module tree splits by responsibility: [`scene`] owns the
//!   floor and facing arrow, [`overlays`] owns the data-driven fat-line
//!   overlays (bounds + interaction + footprint tiles),
//!   [`kind_panel`] owns the left-side egui surface,
//!   [`hot_reload`] owns the file-watcher pump + template rebuild, and
//!   [`mesh_edit`] owns the recalc / Save / auto-frame trio.
//!
//! Controls:
//! - Click a kind in the left panel — swap the displayed item, re-frame the camera.
//! - Right-click drag — rotate the camera (yaw + pitch).
//! - Scroll wheel — zoom.
//! - W / A / S / D — pan the focal point.
//! - Esc — quit.

mod hot_reload;
mod kind_panel;
mod mesh_edit;
mod overlays;
mod scene;
mod sim;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{Definitions, FsSource, KindId, Vfs, VfsPath};
use currawong::glam::{Mat4, UVec2, Vec3, Vec4};
use currawong::{
    AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, Frustum, InstanceBuckets,
    MaterialId, MaterialRegistry, MeshDraw, MeshInstanceAttribs, MeshTemplate, OrbitRig,
    PbrAtlasMaterial, PbrAtlasMaterialInstance, PbrAtlasMaterialParams, PbrAtlasMaterials,
    PbrMaterial, PbrMaterialInstance, RenderObjectTraversal, RenderProxies, RenderRegistry,
    RenderSpec, RenderTemplate, Renderer, SamplerKind, SamplerRegistry, ShadowMeshPipeline,
    SunCascades, TextureColorSpace, View, ViewConfig, ViewEnvironment, ZoneId, egui, pollster,
    wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use overlays::{
    BoundsOverlay, FootprintTilesOverlay, InteractionTilesOverlay, build_bounds_overlay,
    build_footprint_overlay, build_interaction_overlay, write_bounds_instance,
};
use scene::{
    FacingArrowOverlay, GroundPlane, build_facing_arrow_overlay, build_ground_plane,
    write_facing_arrow_instance,
};
use sim::{Command, Game};

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

// --- View --------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Generous enough for any single-item view; only one part draws at a time.
const MAX_INSTANCES_PER_PART: u32 = 4;

/// `PartKey` collapses to `KindId`: each kind is exactly one body part.
pub(crate) type Templates = RenderRegistry<KindId, KindId, KindId>;

pub(crate) struct LumberEditorView {
    pub(crate) camera: Camera,
    pub(crate) camera_binding: CameraBinding,
    pub(crate) rig: OrbitRig,

    pub(crate) material: PbrMaterial,
    pub(crate) atlas_material: PbrAtlasMaterial,
    pub(crate) atlas_materials: MaterialRegistry<PbrAtlasMaterialInstance>,
    pub(crate) samplers: SamplerRegistry,
    pub(crate) asset_server: AssetServer,

    pub(crate) mesh_templates: HashMap<KindId, MeshTemplate<PbrMaterialInstance>>,
    pub(crate) templates: Templates,
    pub(crate) proxies: RenderProxies<KindId>,
    pub(crate) buckets: InstanceBuckets<KindId, MeshInstanceAttribs>,

    /// Depth-only pipeline for the four cascade shadow passes per frame.
    /// Shares the canonical `PosNormalUv` + `MeshInstanceAttribs` layout, so
    /// the same instance buckets we draw in `render` are re-bound under the
    /// depth-only pipeline.
    pub(crate) shadow_pipeline: ShadowMeshPipeline,

    /// Static checkerboard ground plane that catches the kind's shadow.
    /// Single fixed-instance draw issued at the top of `render`; not part of
    /// the proxy/template pipeline because it isn't tied to a sim object.
    pub(crate) ground: GroundPlane,

    /// Yellow wireframe AABB drawn around the selected kind's visual bounds.
    /// Shares one static unit-cube line buffer across kinds; the per-instance
    /// model matrix is re-uploaded on selection change to scale the unit cube
    /// to the kind's AABB.
    pub(crate) bounds_overlay: BoundsOverlay,

    /// Green flat squares drawn on the ground, one per interaction tile
    /// declared by the selected kind's `Interaction`. Same shape as
    /// `bounds_overlay`: one static unit-quad shared across kinds, one
    /// growable per-frame instance buffer holding a `MeshInstanceAttribs`
    /// per tile. Drawn before the bounds overlay so the yellow AABB lines
    /// read on top of the green squares.
    pub(crate) interaction_overlay: InteractionTilesOverlay,

    /// Orange X-marked squares drawn on the ground, one per placement
    /// tile declared by the selected kind's `Footprint`. Same shape as
    /// `interaction_overlay`, with the unit-square geometry augmented by
    /// two diagonals so a single tile reads as a filled placement marker
    /// rather than just an outline.
    pub(crate) footprint_overlay: FootprintTilesOverlay,

    /// Yellow arrow drawn on the ground from the AABB's front face in
    /// the facing direction, 1 m long. Static shaft + arrowhead vertex
    /// buffer; the per-instance model matrix is rewritten each frame
    /// from the selected kind's AABB and `WorldTransform::facing`.
    pub(crate) facing_arrow_overlay: FacingArrowOverlay,

    /// Previous frame's selection; auto-frame fires when this differs from
    /// the sim's current `KindId` component on `subject`.
    pub(crate) last_selected: Option<KindId>,

    /// View-side visibility toggles driven by the left-panel checkboxes.
    /// Each gates the matching draw block in `render` (and the body's
    /// `depth_only` shadow pass so hidden meshes don't leave a shadow).
    pub(crate) show_main_mesh: bool,
    pub(crate) show_bounding_box: bool,
    pub(crate) show_interaction_tiles: bool,
    pub(crate) show_footprint_tiles: bool,
    pub(crate) show_facing_arrow: bool,

    /// Set in [`Self::update`] when the file watcher reports changes and
    /// definitions re-parse cleanly. Drained in [`Self::shadow_pass`]
    /// (cascade 0) to rebuild `mesh_templates` + `templates`. Two-phase
    /// because `update` has no `Renderer` and the template rebuild needs
    /// one — splitting the parse half (file-system reads) from the GPU
    /// half (handle requests, material instances) keeps the seam clean.
    pub(crate) pending_defs: Option<Definitions>,

    /// Source `.ron` file (relative to the VFS root) each kind was loaded
    /// from. Populated alongside the per-kind template caches in `init` and
    /// `maybe_rebuild_templates`. The Save button uses it to find the file
    /// to rewrite.
    pub(crate) kind_sources: HashMap<KindId, VfsPath>,

    /// On-disk root the VFS is mounted on. Joined with a [`VfsPath`] to get
    /// a real `Path` the editor can write to when the user clicks Save.
    pub(crate) assets_root: PathBuf,

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
    pub(crate) pending_edit: Option<(KindId, RenderSpec)>,
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
        self.kind_panel(sim, cmds, egui_ctx);
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

// --- Entry point -------------------------------------------------------

fn main() {
    let vfs = Arc::new(lumber_editor_vfs());
    let kinds_prefix = VfsPath::new("kinds").expect("valid VFS path");
    let defs = pollster::block_on(Definitions::load(&vfs, &kinds_prefix))
        .expect("loading kind definitions");
    let game = Game::new(defs);
    currawong::run::<LumberEditorView>(game);
}
