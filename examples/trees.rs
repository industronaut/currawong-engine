//! Live-sim render template + end-to-end picking showcase.
//!
//! Sim spawns ~200 trees, each carrying a [`Tree`] component (age + seed).
//! Every tick increments age; every frame the view feeds the engine's
//! [`RenderObjectPass`] and runs [`extract_part`] for each alive instance,
//! turning current age into `height` + `tint` for that frame. Trees visibly
//! grow and shift colour over ~12 s at speed 1.0; pausing freezes growth.
//!
//! Adding a new slot (e.g. `sway_phase: F32`) requires only a
//! `with_slot(...)` on the template and a one-liner in [`extract_part`].
//! No edits to the render walk itself — the v1 ergonomics goal from #8.
//!
//! ## Picking (#56)
//!
//! This is the first example that exercises **both** halves of the GPU
//! hit-ID buffer:
//! - Terrain cells are pickable through the engine's per-chunk ID writes
//!   (PR 2). Hovering over the ground highlights the tile and shows its
//!   coordinate in the title bar.
//! - Trees are pickable through per-instance hit-ID writes via
//!   [`MeshInstanceAttribs::with_hit_id`] (PR 3). Hovering over any part
//!   of a tree (trunk or canopy) highlights *that whole tree* and shows
//!   its `WorldObjectId` + current age in the title bar — clicking the
//!   trunk or the canopy resolves to the same sim object.
//!
//! The dispatch from a single [`Renderer::hit_id_hover`] call to the two
//! consumers (terrain → [`CellHighlight`], tree → per-instance tint
//! override) is the v1 reference for any user code that wants to do
//! multi-kind picking.
//!
//! Controls:
//! - Right-click drag — rotate the camera around the focal point.
//! - W / A / S / D — pan the focal point on the ground.
//! - Scroll wheel — zoom.
//! - Mouse over terrain or trees — highlight + title-bar readout.
//! - `0`/`P` pause, `1`/`2`/`3` set sim speed.
//! - `Esc` to quit.
//!
//! Build with `--features egui` for a debug overlay: FPS, a rolling
//! frametime strip chart with a 60 Hz reference line, and sim-clock controls.

use std::collections::HashMap;
#[cfg(feature = "egui")]
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::time::Duration;
#[cfg(feature = "egui")]
use std::time::Instant;

#[cfg(feature = "egui")]
use currawong::egui;
use currawong::glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, CellHighlight, EngineCtx, FlatTopsMesher, Frustum, HitTarget,
    InstanceBuckets, LiveRenderObjects, MaterialInstanceRegistry, MeshInstanceAttribs, MeshPart,
    OrbitRig, RenderObjectPass, RenderRegistry, RenderTemplate, Renderer, Simulation, SlotKey,
    SlotKind, SquareGrid, TerrainMaterial, TerrainMaterialInstance, TerrainPicker, TerrainRenderer,
    TileCoord, UnlitColoredInstance, UnlitColoredMaterial, View, ViewConfig, WorldObjectId,
    WorldTransform, Zone, ZoneId, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ---------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Tree {
    age_ticks: u32,
    tint_seed: f32,
}

const MATURE_AGE: u32 = 720; // ~12 s at the default 60 Hz tick rate.
const TERRAIN_HALF: i32 = 12;
const HEIGHT_UNIT: f32 = 0.1;
const TILE_SIZE: f32 = 1.0;
const TREE_COUNT: u32 = 200;
const BASE_TITLE: &str = "currawong — trees demo";

// Tint applied to whichever tree is under the cursor, replacing the
// per-instance leaf/trunk tint for that one frame. Hot orange reads well
// against the green canopy palette without clobbering depth cues.
const HOVER_TINT: Vec4 = Vec4::new(1.0, 0.55, 0.10, 1.0);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RenderId {
    TreeOak,
}

struct Game {
    zones: Zones,
    main_zone: ZoneId,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let main_zone = zones.insert(Zone::new());
        let zone = zones.get_mut(main_zone).expect("just inserted");

        let terrain = zone.terrain_mut();
        for ty in -TERRAIN_HALF..TERRAIN_HALF {
            for tx in -TERRAIN_HALF..TERRAIN_HALF {
                let h = ((tx as f32 * 0.4).sin() * 2.0 + (ty as f32 * 0.4).cos() * 1.5).round();
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = h as i32;
            }
        }

        // Fixed seed → identical layout every run.
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..TREE_COUNT {
            let tx = rng.range_i32(-TERRAIN_HALF, TERRAIN_HALF);
            let ty = rng.range_i32(-TERRAIN_HALF, TERRAIN_HALF);
            let floor_h = zone
                .terrain()
                .tile_or_default(TileCoord::new(tx, ty))
                .floor_height as f32
                * HEIGHT_UNIT;
            let position = Vec3::new(
                tx as f32 + 0.5 + rng.next_f32() - 0.5,
                ty as f32 + 0.5 + rng.next_f32() - 0.5,
                floor_h,
            );
            let tint_seed = rng.next_f32();
            let rotation = Quat::from_rotation_z(rng.next_f32() * TAU);
            let id = zone.insert(WorldTransform { position, rotation });
            zone.components_mut().insert(id, RenderId::TreeOak);
            zone.components_mut().insert(
                id,
                Tree {
                    age_ticks: 0,
                    tint_seed,
                },
            );
        }
        Self { zones, main_zone }
    }
}

impl Simulation for Game {
    fn tick(&mut self, _: Duration) {
        for (_, zone) in self.zones.iter_mut() {
            // Snapshot ids first — `get_mut` would conflict with the iter borrow.
            let ids: Vec<_> = zone.components().iter::<Tree>().map(|(id, _)| id).collect();
            for id in ids {
                if let Some(tree) = zone.components_mut().get_mut::<Tree>(id) {
                    tree.age_ticks = tree.age_ticks.saturating_add(1);
                }
            }
        }
    }
}

// --- Tiny deterministic RNG --------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u32() % (hi - lo) as u32) as i32
    }
}

// --- View -------------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 512;
#[cfg(feature = "egui")]
const FRAMETIME_HISTORY: usize = 240; // ~4 s at 60 fps.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MeshHandle {
    Trunk,
    Canopy,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MatKey {
    Bark,
    Leaf,
}

type Templates = RenderRegistry<RenderId, MeshHandle, MatKey>;

struct Demo {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,
    /// Terrain-side picker. Drives both the zero-latency ray-vs-plane
    /// fallback and the GPU ID-buffer terrain hover. Mesh-object hits
    /// bypass this and land in [`Self::hovered_object`] instead.
    picker: TerrainPicker<SquareGrid>,
    /// Per-frame yellow overlay over the hovered terrain tile.
    highlight: CellHighlight,
    /// Which tree (if any) the cursor is currently over — driven by the
    /// engine's `HitTarget::Object` resolution this frame. The extract
    /// loop swaps in [`HOVER_TINT`] for whichever tree matches.
    hovered_object: Option<WorldObjectId>,
    /// Cache of the last title text we sent to winit, so we only call
    /// `set_title` when the readout actually changes.
    last_title: Option<String>,
    material: UnlitColoredMaterial,
    instances: MaterialInstanceRegistry<UnlitColoredInstance, MatKey>,
    templates: Templates,
    live_objects: LiveRenderObjects<RenderId>,
    meshes: HashMap<MeshHandle, GpuMesh>,
    buckets: InstanceBuckets<MeshHandle, MeshInstanceAttribs>,
    terrain_material: TerrainMaterial,
    terrain_solid: TerrainMaterialInstance,
    terrain: TerrainRenderer,
    last_draws: u32,
    #[cfg(feature = "egui")]
    last_frame: Instant,
    #[cfg(feature = "egui")]
    frame_samples: VecDeque<f32>,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl View for Demo {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: BASE_TITLE,
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;
        let camera_binding = CameraBinding::new(device);
        let material = UnlitColoredMaterial::new(renderer, camera_binding.layout());

        // Orbit rig with the focal point parked over the origin; pitch +
        // distance chosen so the whole tree field is visible at startup,
        // and the user can pan WASD to put any specific tree under the
        // cursor for picking.
        let mut rig = OrbitRig::new(Vec3::new(0.0, 0.0, 1.0));
        rig.distance = 18.0;
        rig.pitch = 45.0_f32.to_radians();
        rig.config.distance_max = 80.0;

        let picker = TerrainPicker::new(SquareGrid, TILE_SIZE);
        // Warm yellow at 50% alpha for the hovered terrain cell — same
        // colour the terrain demo uses, so the picker readout is visually
        // consistent across examples.
        let highlight = CellHighlight::new(
            renderer,
            camera_binding.layout(),
            Vec4::new(1.0, 0.85, 0.2, 0.5),
        );

        let mut instances = MaterialInstanceRegistry::new();
        instances.register(
            MatKey::Bark,
            material.create_instance(renderer, Vec4::new(0.42, 0.28, 0.16, 1.0)),
        );
        // Canopy base is white; per-instance tint carries young → mature ramp + jitter.
        instances.register(MatKey::Leaf, material.create_instance(renderer, Vec4::ONE));

        // Unit tree, root at z=0: trunk 0..0.4, canopy 0.3..1.0 in template Z.
        // The `height` slot Z-scales the whole template per-instance; XY stays put.
        let mut templates: Templates = RenderRegistry::new();
        templates.register(
            RenderId::TreeOak,
            RenderTemplate::new("tree_oak")
                .with_slot("height", SlotKind::F32)
                .with_slot("tint", SlotKind::Color)
                .with_mesh_part(
                    MeshHandle::Trunk,
                    MatKey::Bark,
                    Mat4::from_scale(Vec3::new(0.06, 0.06, 0.4)),
                )
                .with_mesh_part(
                    MeshHandle::Canopy,
                    MatKey::Leaf,
                    Mat4::from_scale_rotation_translation(
                        Vec3::new(0.45, 0.45, 0.7),
                        Quat::IDENTITY,
                        Vec3::new(0.0, 0.0, 0.3),
                    ),
                ),
        );

        let mut meshes = HashMap::new();
        meshes.insert(MeshHandle::Trunk, upload_mesh(device, &cylinder(8)));
        meshes.insert(MeshHandle::Canopy, upload_mesh(device, &cone(8)));

        let mut buckets = InstanceBuckets::new("tree attribs", MAX_INSTANCES);
        for mesh in [MeshHandle::Trunk, MeshHandle::Canopy] {
            buckets.register(device, mesh);
        }

        let terrain_material = TerrainMaterial::new(renderer, camera_binding.layout());
        let terrain_solid = terrain_material.create_instance(renderer, Vec4::ONE);

        Self {
            camera: Camera::default(),
            camera_binding,
            rig,
            picker,
            highlight,
            hovered_object: None,
            last_title: None,
            material,
            instances,
            templates,
            live_objects: LiveRenderObjects::new(30),
            meshes,
            buckets,
            terrain_material,
            terrain_solid,
            terrain: TerrainRenderer::new(),
            last_draws: 0,
            #[cfg(feature = "egui")]
            last_frame: Instant::now(),
            #[cfg(feature = "egui")]
            frame_samples: VecDeque::with_capacity(FRAMETIME_HISTORY),
        }
    }

    fn update(&mut self, _: &Game, _: &mut EngineCtx, dt: Duration) {
        // Integrate held-WASD pan + held rotation against wall-clock dt so
        // panning keeps working at sim speed 0 — see camera_rig docs.
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);
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
        self.camera.far = 200.0;
        self.camera_binding.write(&renderer.queue, &self.camera);

        let zone = sim.zones.get(sim.main_zone).expect("main zone");
        if self.terrain.is_empty() {
            self.terrain.rebuild_all(
                renderer,
                zone.terrain(),
                &FlatTopsMesher {
                    tile_size: TILE_SIZE,
                    height_unit: HEIGHT_UNIT,
                    ..FlatTopsMesher::new()
                },
            );
        }

        // --- Picker dispatch ---------------------------------------------
        //
        // One GPU readback feeds two consumers this frame: the terrain
        // hover lands in `TerrainPicker::id_hover` (which prefers it over
        // the ray-plane fallback for slope correctness), and an object hit
        // lands in `self.hovered_object` so the extract loop below can
        // swap in the highlight tint for that one tree. Hand-rolling this
        // dispatch — instead of having a single picker handle both — is
        // the v1 reference for any user code with more than one kind of
        // pickable.
        let viewport = Vec2::new(size.width as f32, size.height as f32);
        self.picker.update(&self.camera, viewport);
        let hit_hover = renderer.hit_id_hover();
        self.hovered_object = match hit_hover {
            Some(HitTarget::Object { id, .. }) => Some(id),
            _ => None,
        };
        // The picker filters internally: TerrainCell populates id_hover,
        // anything else clears it.
        self.picker.set_id_hover(hit_hover);

        // --- Title readout -----------------------------------------------
        //
        // Tree hover takes precedence over terrain hover — when the user's
        // cursor sits on a tree the cell underneath isn't what they're
        // pointing at. Includes the tree's current age so the live-sim
        // growth story stays visible even at rest.
        let hover_cell = self
            .picker
            .hover()
            .map(|h| TileCoord::new(h.cell.x, h.cell.y));
        let title = match (self.hovered_object, hover_cell) {
            (Some(id), _) => {
                let age = zone
                    .components()
                    .get::<Tree>(id)
                    .map(|t| t.age_ticks)
                    .unwrap_or(0);
                format!("{BASE_TITLE} — tree #{} (age {age} ticks)", id.index())
            }
            (None, Some(c)) => format!("{BASE_TITLE} — cell ({}, {})", c.x, c.y),
            (None, None) => BASE_TITLE.to_string(),
        };
        if self.last_title.as_deref() != Some(title.as_str()) {
            renderer.window.set_title(&title);
            self.last_title = Some(title);
        }

        // Drive the terrain overlay only when the cursor is on the ground,
        // not when it's parked on a tree. The Z hugs the visible floor
        // height (looked up from sim) plus a hair, so the overlay tracks
        // the mesh top across slopes.
        match (self.hovered_object, hover_cell) {
            (None, Some(coord)) => {
                let h = zone.terrain().tile_or_default(coord).floor_height;
                let z = h as f32 * HEIGHT_UNIT + 0.02;
                self.highlight
                    .set_cell(renderer, &SquareGrid, coord, z, TILE_SIZE);
            }
            _ => self.highlight.clear(),
        }

        // --- Render-object pass ------------------------------------------

        let frustum = Frustum::from_view_proj(self.camera.view_proj());
        RenderObjectPass::declare_and_cull(
            &sim.zones,
            &self.templates,
            &mut self.live_objects,
            &frustum,
        );

        // Per-frame slot extraction loop. The whole slot-driven mapping
        // (age → height → world matrix; age + seed → tint) lives inside
        // `extract_part`; adding a new slot is local to that function.
        //
        // A single GPU hit ID per tree is reserved here (#56 PR 3) and
        // stamped onto every mesh part of that tree, so a cursor over the
        // trunk or canopy resolves to the same `WorldObjectId` through
        // `renderer.hit_id_hover()` on the next-but-one frame. The
        // hovered tree gets its per-instance tint overridden by
        // `HOVER_TINT` so the visual feedback matches the title readout.
        self.buckets.begin_frame();
        for (parent, rid, object) in self.live_objects.iter() {
            let tree = sim
                .zones
                .get(parent.zone)
                .and_then(|z| z.components().get::<Tree>(parent.id))
                .copied()
                .unwrap_or_default();
            let template = self.templates.get(rid).expect("declared template");
            let hit_id = renderer.reserve_object(parent.zone, parent.id);
            let is_hovered = Some(parent.id) == self.hovered_object;
            for part in template.mesh_parts() {
                let mut attribs = extract_part(&tree, part, object.world_xform);
                if is_hovered {
                    attribs.tint = HOVER_TINT.to_array();
                }
                self.buckets.push(part.mesh, attribs.with_hit_id(hit_id));
            }
        }
        self.buckets.upload(&renderer.queue);

        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.main_zone, &self.terrain_solid);
        // Every chunk in this example has a solid mesh (all tiles set their
        // floor_height), so chunk_count is the exact terrain draw total.
        let mut draws = self.terrain.chunk_count() as u32;

        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        for (mesh_handle, instance_buffer, count) in self.buckets.iter_filled() {
            let mesh = &self.meshes[mesh_handle];
            let mat_key = match mesh_handle {
                MeshHandle::Trunk => MatKey::Bark,
                MeshHandle::Canopy => MatKey::Leaf,
            };
            pass.set_bind_group(
                1,
                self.instances
                    .get(mat_key)
                    .expect("registered")
                    .bind_group(),
                &[],
            );
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
            draws += 1;
        }

        // Terrain hover outline last so it lands on top of solids; no-op
        // when nothing is hovered.
        self.highlight.draw(pass);
        self.last_draws = draws;
    }

    fn input(&mut self, _: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
        // Both helpers see every event. The rig consumes mouse-buttons /
        // scroll / WASD; the picker filters to CursorMoved / CursorLeft.
        // Each ignores what it doesn't care about, so order doesn't matter.
        self.rig.handle_event(event);
        self.picker.handle_event(event);

        let WindowEvent::KeyboardInput { event, .. } = event else {
            return;
        };
        if event.state != ElementState::Pressed {
            return;
        }
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        match code {
            KeyCode::Escape => ctx.event_loop.exit(),
            KeyCode::Digit0 | KeyCode::KeyP => ctx.clock.set_speed(0.0),
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            KeyCode::Digit2 => ctx.clock.set_speed(2.0),
            KeyCode::Digit3 => ctx.clock.set_speed(3.0),
            _ => {}
        }
    }

    #[cfg(feature = "egui")]
    fn ui(&mut self, _: &mut Game, ctx: &mut EngineCtx, egui_ctx: &egui::Context) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_samples.len() == FRAMETIME_HISTORY {
            self.frame_samples.pop_front();
        }
        self.frame_samples.push_back(dt);
        let n = self.frame_samples.len().max(1);
        let avg = self.frame_samples.iter().sum::<f32>() / n as f32;
        let max = self.frame_samples.iter().copied().fold(0.0f32, f32::max);
        let fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };

        egui::Window::new("debug")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(egui_ctx, |ui| {
                ui.label(format!(
                    "fps: {fps:5.1}   avg {:.2} ms   max {:.2} ms",
                    avg * 1000.0,
                    max * 1000.0
                ));
                draw_frametime_chart(ui, &self.frame_samples, max);
                ui.label(format!("draws: {}", self.last_draws));
                ui.separator();
                ui.label(format!("sim ticks: {}", ctx.clock.total_ticks()));
                let mut speed = ctx.clock.speed();
                ui.horizontal(|ui| {
                    let label = if ctx.clock.is_paused() {
                        "play"
                    } else {
                        "pause"
                    };
                    if ui.button(label).clicked() {
                        let new = if ctx.clock.is_paused() { 1.0 } else { 0.0 };
                        ctx.clock.set_speed(new);
                        speed = new;
                    }
                    ui.add(egui::Slider::new(&mut speed, 0.0..=4.0).text("speed"));
                });
                if (speed - ctx.clock.speed()).abs() > f32::EPSILON {
                    ctx.clock.set_speed(speed);
                }
            });
    }
}

/// Rolling frametime strip chart drawn straight into the egui window via
/// `Painter` — no extra deps. The 16.6 ms (60 Hz) line is the budget;
/// samples poking above it are dropped frames.
#[cfg(feature = "egui")]
fn draw_frametime_chart(ui: &mut egui::Ui, samples: &VecDeque<f32>, max_sample: f32) {
    const REF_DT: f32 = 1.0 / 60.0;
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(22));

    // Y scale: 1.2× the higher of the worst sample and the 60 Hz budget,
    // so the reference line is always inside the chart.
    let y_max = (max_sample.max(REF_DT) * 1.2).max(1e-4);
    let y_for = |dt: f32| {
        let n = (dt / y_max).clamp(0.0, 1.0);
        rect.bottom() - n * rect.height()
    };

    let y_ref = y_for(REF_DT);
    painter.line_segment(
        [
            egui::pos2(rect.left(), y_ref),
            egui::pos2(rect.right(), y_ref),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 130, 80)),
    );

    if samples.len() >= 2 {
        let last_idx = (samples.len() - 1) as f32;
        let pts: Vec<egui::Pos2> = samples
            .iter()
            .enumerate()
            .map(|(i, dt)| {
                let x = rect.left() + (i as f32 / last_idx) * rect.width();
                egui::pos2(x, y_for(*dt))
            })
            .collect();
        painter.add(egui::Shape::line(
            pts,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 220, 240)),
        ));
    }

    // Labels sit in low-traffic regions of the chart so the trace doesn't run
    // through them: y-scale bounds in the corners, "60 Hz" just *below* the
    // reference line on the left (samples above the line are spikes, samples
    // below are rare on vsync — so the under-line area stays clear).
    let label_font = egui::FontId::monospace(9.0);
    painter.text(
        rect.left_top() + egui::vec2(3.0, 1.0),
        egui::Align2::LEFT_TOP,
        format!("{:.1} ms", y_max * 1000.0),
        label_font.clone(),
        egui::Color32::from_gray(210),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(3.0, -1.0),
        egui::Align2::LEFT_BOTTOM,
        "0 ms",
        label_font.clone(),
        egui::Color32::from_gray(140),
    );
    painter.text(
        egui::pos2(rect.left() + 3.0, y_ref + 1.0),
        egui::Align2::LEFT_TOP,
        "60 Hz",
        label_font,
        egui::Color32::from_rgb(150, 200, 150),
    );
}

/// Slot-extraction closure for one mesh part of one tree. Owns the
/// (sim state → slot value → per-instance attrib) mapping; nothing in the
/// per-frame walk above knows about `Tree`.
fn extract_part(
    tree: &Tree,
    part: &MeshPart<MeshHandle, MatKey>,
    root: Mat4,
) -> MeshInstanceAttribs {
    let maturity = smoothstep((tree.age_ticks as f32 / MATURE_AGE as f32).clamp(0.0, 1.0));
    // `height` slot: 0.3 → 3.0 along Z, applied between root and part transform
    // so the trunk's XY radius stays put while the tree grows upward.
    let height = 0.3 + 2.7 * maturity;
    let world = root * Mat4::from_scale(Vec3::new(1.0, 1.0, height)) * part.local_transform;

    // `tint` slot: young → mature green plus per-seed jitter. Only the
    // leaf material consumes it; bark stays flat brown.
    let young = Vec4::new(0.55, 0.85, 0.30, 1.0);
    let mature = Vec4::new(0.15, 0.42, 0.18, 1.0);
    let s = tree.tint_seed;
    let jitter = Vec4::new(
        (s * TAU).sin() * 0.06,
        (s * 4.10 + 1.7).sin() * 0.06,
        (s * 9.31 + 2.3).sin() * 0.06,
        0.0,
    );
    let tint = (young.lerp(mature, maturity) + jitter)
        .max(Vec4::ZERO)
        .min(Vec4::ONE)
        .with_w(1.0);
    let part_tint = match part.material {
        MatKey::Leaf => tint,
        MatKey::Bark => Vec4::ONE,
    };
    MeshInstanceAttribs::new(world, part_tint)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// --- Procedural geometry ----------------------------------------------------
// Unit cylinder/cone in template-local space: radius 1, z=0..1.

struct MeshCpu {
    positions: Vec<[f32; 3]>,
    indices: Vec<u16>,
}

fn cylinder(sides: u16) -> MeshCpu {
    let mut positions = Vec::with_capacity(sides as usize * 2 + 2);
    for z in [0.0, 1.0] {
        for i in 0..sides {
            let a = i as f32 / sides as f32 * TAU;
            positions.push([a.cos(), a.sin(), z]);
        }
    }
    let bottom_centre = positions.len() as u16;
    positions.push([0.0, 0.0, 0.0]);
    let top_centre = positions.len() as u16;
    positions.push([0.0, 0.0, 1.0]);

    let mut indices = Vec::new();
    for i in 0..sides {
        let j = (i + 1) % sides;
        indices.extend_from_slice(&[i, i + sides, j + sides, i, j + sides, j]);
        indices.extend_from_slice(&[top_centre, i + sides, j + sides]);
        indices.extend_from_slice(&[bottom_centre, j, i]);
    }
    MeshCpu { positions, indices }
}

fn cone(sides: u16) -> MeshCpu {
    let mut positions = Vec::with_capacity(sides as usize + 2);
    for i in 0..sides {
        let a = i as f32 / sides as f32 * TAU;
        positions.push([a.cos(), a.sin(), 0.0]);
    }
    let tip = positions.len() as u16;
    positions.push([0.0, 0.0, 1.0]);
    let base_centre = positions.len() as u16;
    positions.push([0.0, 0.0, 0.0]);

    let mut indices = Vec::new();
    for i in 0..sides {
        let j = (i + 1) % sides;
        indices.extend_from_slice(&[i, j, tip, base_centre, j, i]);
    }
    MeshCpu { positions, indices }
}

fn upload_mesh(device: &wgpu::Device, mesh: &MeshCpu) -> GpuMesh {
    use wgpu::util::DeviceExt;
    GpuMesh {
        vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tree mesh positions"),
            contents: bytemuck::cast_slice(&mesh.positions),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tree mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }),
        index_count: mesh.indices.len() as u32,
    }
}

fn main() {
    currawong::run::<Demo>(Game::new());
}
