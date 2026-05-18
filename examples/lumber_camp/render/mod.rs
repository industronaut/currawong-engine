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
//! Sim-vocabulary slot names + view-state-on-`LiveRenderObject` from
//! [`super`]/CLAUDE.md still apply: the marker and the carried log are
//! view-side visibility decisions on the persistent
//! [`LiveRenderObject`], driven by the sim's typed [`Designated`] /
//! [`Carrying`] components.

mod pawn;
mod tree;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{Definitions, KindId, VfsPath};
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Camera, CameraBinding, EngineCtx, FlatTopsMesher, Frustum, Handle,
    HitTarget, InstanceBuckets, LiveRenderObjects, MeshInstanceAttribs, OrbitRig, PbrMaterial,
    PbrMaterialInstance, PbrMaterialParams, PrimitiveMesh, RenderObjectPass, RenderRegistry,
    RenderTemplate, Renderer, SamplerKind, SamplerRegistry, TerrainMaterial,
    TerrainMaterialInstance, TerrainRenderer, Texture, TextureColorSpace, View, ViewConfig,
    ViewEnvironment, WorldObjectRef, ZoneId, wgpu, winit, yakui,
};
use serde::Deserialize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use pawn::PawnRenderer;

use crate::sim::{Game, GameState, HEIGHT_UNIT, TILE_SIZE, TIME_LIMIT_SECS, WOOD_GOAL};

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
    /// the kind's `render` block.
    Body(KindId),
    /// Shared red-cone marker shown above designated trees.
    Marker,
    /// Shared brown-cylinder log shown on carrying pawns.
    CarriedLog,
}

impl PartKey {
    /// Body parts receive the hover tint; ancillary parts don't (keeps the
    /// marker readably red, the carried log readably brown, no matter
    /// which body it sits on).
    fn is_body(&self) -> bool {
        matches!(self, Self::Body(_))
    }
}

// --- Per-template GPU resources -----------------------------------------

/// Mesh buffers + the PBR material instance to draw them with. One per
/// [`PartKey`]; bound together because the draw loop always swaps both at
/// once.
///
/// `mesh` is either a [`Handle<Mesh>`] (the kind's body, streamed from
/// the def's `render.mesh` path) or an inline `(vertex, index)` buffer
/// pair (the procedural ancillary parts). Both end up resolved to
/// `wgpu::Buffer` slices at draw time via [`Self::resolve`], so the
/// extract / draw loop is mesh-source-agnostic.
pub struct MeshTemplate {
    pub mesh: MeshSource,
    pub visual_bounds: Aabb,
    pub material: PbrMaterialInstance,
}

pub enum MeshSource {
    /// glTF body part — buffers live behind a streaming handle and we
    /// pay the per-frame `resolve_mesh` to surface them.
    Streamed { handle: Handle<currawong::Mesh> },
    /// Procedural ancillary part — a single owned [`currawong::MeshPrimitive`]
    /// wrapped in a `Vec` so `ResolvedDraw::primitives` is one shape
    /// regardless of source. No streaming, no fallback.
    Inline {
        primitives: Vec<currawong::MeshPrimitive>,
    },
}

/// Per-draw resolution of a [`MeshTemplate`] into a slice of primitives
/// plus the model-matrix adjustment the caller composes inside their
/// per-instance world transform. For streamed templates this matches
/// [`AssetServer::resolve_mesh`] exactly; for inline templates the slice
/// is a single primitive borrowed from a per-template scratchpad and the
/// adjustment is identity.
pub struct ResolvedDraw<'a> {
    pub primitives: &'a [currawong::MeshPrimitive],
    pub fallback_adjustment: Mat4,
}

impl MeshTemplate {
    pub fn resolve<'a>(&'a self, asset_server: &'a AssetServer) -> ResolvedDraw<'a> {
        match &self.mesh {
            MeshSource::Streamed { handle } => {
                let r = asset_server.resolve_mesh(handle, Some(self.visual_bounds));
                ResolvedDraw {
                    primitives: r.primitives,
                    fallback_adjustment: r.fallback_adjustment,
                }
            }
            MeshSource::Inline { primitives } => ResolvedDraw {
                primitives,
                fallback_adjustment: Mat4::IDENTITY,
            },
        }
    }
}

// --- Def deserialisation -----------------------------------------------

/// The view-side projection of each kind def's `render` block. The sim
/// has its own per-kind body structs that pick out the sim-relevant
/// fields — serde silently drops the rest, so the two views stay
/// independent.
#[derive(Debug, Clone, Deserialize)]
pub struct RenderSpec {
    /// One of the closed set `"tree" | "pawn" | "building"`. Drives which
    /// view-side factory builds the template.
    pub shape: String,
    pub mesh: String,
    pub albedo: String,
    pub metallic: f32,
    pub roughness: f32,
    pub bounds_min: (f32, f32, f32),
    pub bounds_max: (f32, f32, f32),
}

/// Every kind in the lumber-camp content has a `render` block — it
/// exists because the sim wants to draw the kind. A future "rules-only"
/// kind (e.g. a recipe, a faction marker) would have no `render`; for
/// PR4 we make it required and surface a clear parse error if it's
/// missing rather than silently dropping the kind from the world.
#[derive(Deserialize)]
struct KindDefBody {
    render: RenderSpec,
}

impl RenderSpec {
    pub fn visual_bounds(&self) -> Aabb {
        Aabb::new(
            Vec3::new(self.bounds_min.0, self.bounds_min.1, self.bounds_min.2),
            Vec3::new(self.bounds_max.0, self.bounds_max.1, self.bounds_max.2),
        )
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
    samplers: SamplerRegistry,
    asset_server: AssetServer,

    /// GPU bundle per drawable part. Looked up by the draw loop after the
    /// per-part bucket has been filled by the engine walk. Refresh runs
    /// each frame inside [`Self::render`] to reconcile the texture handles
    /// (cheap when nothing changed, swaps the bind group on the frame the
    /// real PNG lands).
    mesh_templates: HashMap<PartKey, MeshTemplate>,
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
    live_objects: LiveRenderObjects<KindId>,
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
    /// `instance.world_xform`.
    pawn: PawnRenderer,
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

        // View-side VFS — independent of the one main.rs handed to the
        // sim's `Definitions`. Same on-disk content, separate cache.
        let vfs = Arc::new(crate::lumber_camp_vfs());
        let asset_server = AssetServer::new(renderer, vfs.clone());

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
        let mut mesh_templates: HashMap<PartKey, MeshTemplate> = HashMap::new();
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
        for (kind_id, def) in defs.iter() {
            let body: KindDefBody = match def.value.clone().into_rust() {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("lumber_camp: skipping {kind_id}: {e}");
                    continue;
                }
            };
            let render = body.render;
            let Some(shape) = RenderShape::from_tag(&render.shape) else {
                eprintln!(
                    "lumber_camp: kind {kind_id} declares unknown render.shape `{}`; skipping",
                    render.shape
                );
                continue;
            };
            let body_template = build_body_template(
                renderer,
                &material,
                &samplers,
                &asset_server,
                kind_id,
                &render,
            );
            mesh_templates.insert(PartKey::Body(kind_id.clone()), body_template);
            let template = shape.register_template(kind_id.clone(), &render);
            templates.register(kind_id.clone(), template);
            shapes.insert(kind_id.clone(), shape);
        }

        let live_objects = LiveRenderObjects::<KindId>::new(CULL_HYSTERESIS_FRAMES);

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

        Self {
            camera,
            camera_binding,
            rig,
            material,
            samplers,
            asset_server,
            mesh_templates,
            templates,
            shapes,
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
        // object carrying a `KindId` matching a registered template;
        // frustum-culled with hysteresis.
        let frustum = Frustum::from_view_proj(self.camera.view_proj());
        RenderObjectPass::declare_and_cull(
            &sim.zones,
            &self.templates,
            &mut self.live_objects,
            &frustum,
        );

        // Phase 1.5: reconcile material handles + cache the per-template
        // fallback adjustments. Material `refresh` is cheap when nothing
        // changed; on the frame a streamed texture transitions Loading →
        // Ready the bind group is rebuilt against the real view.
        // `resolve_mesh` follows the same shape for meshes (and returns
        // a sizing matrix while the real glTF is still in flight).
        let asset_server = &self.asset_server;
        let material = &self.material;
        let samplers = &self.samplers;
        let mut adjustments: HashMap<PartKey, Mat4> = HashMap::new();
        for (key, template) in &mut self.mesh_templates {
            template
                .material
                .refresh(renderer, material, samplers, asset_server);
            adjustments.insert(
                key.clone(),
                template.resolve(asset_server).fallback_adjustment,
            );
        }

        // Phase 1.7: the single sim→view translation seam. Each
        // RenderShape's update logic lives in its own kind module —
        // dispatching on the precomputed `shapes` map keeps the per-
        // frame seam to one HashMap lookup.
        let pawn = &self.pawn;
        let shapes = &self.shapes;
        RenderObjectPass::update_instances(
            &sim.zones,
            &self.templates,
            &mut self.live_objects,
            |parent, kind_id, _slots, components, instance| match shapes.get(kind_id) {
                Some(RenderShape::Pawn) => {
                    pawn::update_instance(parent, components, instance, alpha, pawn)
                }
                Some(RenderShape::Tree) => tree::update_instance(parent, components, instance),
                // Building: no per-instance view-state mutation; defaults
                // (all parts visible, world_xform from the engine
                // declare) stand. Same for unknown shapes that slipped
                // through (registration would have rejected them).
                Some(RenderShape::Building) | None => {}
            },
        );

        // Phase 2: engine-driven extract. The hit-ID-aware variant reserves
        // one ID per parent so clicking the log resolves back to the pawn,
        // and clicking the marker resolves back to the tree (#56 PR 3).
        // Hover tint stays on body parts; ancillaries keep their own
        // constant albedo for legibility.
        let buckets = &mut self.buckets;
        let hovered = self.hovered;
        RenderObjectPass::for_each_alive_part_with_hit_id(
            &sim.zones,
            &self.templates,
            &self.live_objects,
            renderer,
            |parent, _kind, part, world, _slots, hit_id| {
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

        // Terrain first so opaque ground is in the depth buffer before meshes
        // draw on top of it. Same camera + scene env bindings serve both.
        // No liquids in the skeleton — skip the transparent pass.
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.zone, &self.terrain_solid);

        // Mesh parts: one indexed-instanced call per non-empty bucket,
        // resolving the mesh buffers fresh each draw (cheap — HashMap
        // lookup + Handle::peek). Swapping pipeline once is enough
        // because every kind shares the same PBR material template.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        for (part_key, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(template) = self.mesh_templates.get(part_key) else {
                continue;
            };
            let resolved = template.resolve(asset_server);
            pass.set_bind_group(2, template.material.bind_group(), &[]);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            // Lumber camp's existing assets are all single-primitive, but
            // post-#80 every glb produces a `Vec<MeshPrimitive>`. Loop
            // over them so a future multi-primitive species body draws
            // through the same path without further plumbing.
            for prim in resolved.primitives {
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..count);
            }
        }
    }

    fn input(&mut self, sim: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
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

/// Build the body [`MeshTemplate`] for a kind: streamed glTF mesh +
/// streamed PNG albedo, sized to the def's bounds. Used by every kind
/// regardless of shape — the structural difference between shapes is
/// in the [`RenderTemplate`], not the body template.
fn build_body_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    kind_id: &KindId,
    render: &RenderSpec,
) -> MeshTemplate {
    let mesh_path = VfsPath::new(render.mesh.clone())
        .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.mesh path: {e}"));
    let albedo_path = VfsPath::new(render.albedo.clone())
        .unwrap_or_else(|e| panic!("kind {kind_id}: invalid render.albedo path: {e}"));
    let mesh_handle = asset_server.mesh(mesh_path);
    let albedo_handle = asset_server.texture(albedo_path, TextureColorSpace::Srgb);
    let material_instance = material.create_instance(
        renderer,
        samplers,
        asset_server,
        PbrMaterialParams {
            albedo: albedo_handle,
            sampler: SamplerKind::LinearRepeat,
            // Texture sample carries the colour; per-instance tint
            // multiplier (white) leaves it unchanged unless hover is
            // overriding.
            albedo_factor: Vec4::ONE,
            metallic: render.metallic,
            roughness: render.roughness,
        },
    );
    MeshTemplate {
        mesh: MeshSource::Streamed {
            handle: mesh_handle,
        },
        visual_bounds: render.visual_bounds(),
        material: material_instance,
    }
}

/// Parameters describing an inline (procedural) ancillary part — bundled
/// into a struct because there are enough of them that a positional arg
/// list crosses clippy's `too_many_arguments` threshold.
pub struct InlineTemplate<'a> {
    pub label: &'static str,
    pub mesh: &'a PrimitiveMesh,
    pub bounds: Aabb,
    /// Flat colour multiplier — ancillaries don't stream a texture, so
    /// this is the only place their colour comes from.
    pub albedo_factor: Vec4,
    pub metallic: f32,
    pub roughness: f32,
}

/// Build an inline [`MeshTemplate`] from a [`PrimitiveMesh`] + flat
/// albedo factor. Shared helper for the two procedural ancillary parts
/// (marker, carried log) that don't go through the asset pipeline — they
/// still plug into the same PBR material surface the streamed bodies
/// use, via a 1×1 white texture wrapped in a ready [`Handle`].
pub fn new_inline_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
    params: InlineTemplate<'_>,
) -> MeshTemplate {
    use wgpu::util::DeviceExt;
    let vertex_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(params.label),
            contents: bytemuck::cast_slice(&params.mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let index_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(params.label),
            contents: bytemuck::cast_slice(&params.mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
    let primitive = currawong::MeshPrimitive {
        vertex_buffer,
        index_buffer,
        index_count: params.mesh.index_count(),
        material_name: None,
    };
    let white = Texture::from_rgba8(renderer, "lumber-camp ancillary", 1, 1, &[255; 4], true);
    let material_instance = material.create_instance(
        renderer,
        samplers,
        asset_server,
        PbrMaterialParams {
            albedo: Handle::ready(white),
            sampler: SamplerKind::LinearClamp,
            albedo_factor: params.albedo_factor,
            metallic: params.metallic,
            roughness: params.roughness,
        },
    );
    MeshTemplate {
        mesh: MeshSource::Inline {
            primitives: vec![primitive],
        },
        visual_bounds: params.bounds,
        material: material_instance,
    }
}

/// Format `seconds` as `M:SS` for the HUD countdown.
fn format_clock(seconds: f32) -> String {
    let total = seconds.max(0.0) as u32;
    format!("{}:{:02}", total / 60, total % 60)
}
