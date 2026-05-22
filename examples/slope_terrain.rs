//! Square-grid terrain meshed with [`SlopeMesher`] — each cell's corner
//! sits at `max(floor_height)` of every cell touching it, so neighbouring
//! tiles slope into each other instead of stepping.
//!
//! Layout: a 16×16 patch of ground with a broad rolling hill in the +X +Y
//! quadrant (heights 0–4, falling off with distance from the hill centre)
//! and a small pond on the opposite side. With the flat-shaded fan
//! triangulation, each pair of triangles in a sloped cell catches the sun
//! at a slightly different angle — the surface reads as a faceted Transport
//! Tycoon / Rise of Industry style rather than a Minecraft step pattern.
//!
//! Uses `height_unit = 1.0` (same as `tile_size`) so each integer height
//! step makes a 45° slope across one cell — the canonical Transport Tycoon
//! aspect ratio. Slopes shallower than ~25° tend to wash out under the
//! orbit camera because the lit-side variation runs into the sRGB target's
//! [0, 1] clipping range.
//!
//! Controls:
//! - Right-click drag — rotate yaw / pitch around the focal point.
//! - W / A / S / D — pan the focal point on the ground.
//! - Scroll wheel — zoom (orbit distance).
//! - Y / X — toggle invert-Y / invert-X on the drag rotation.
//! - 0 pause, 1/2/3 sim speed, Esc to quit.
//!
//! Hovered cell appears in the window title — sourced from the engine's
//! GPU hit-ID buffer (#56 PR 2), so on the rolling hill it reports the
//! cell whose visible mesh top is actually under the cursor, not the
//! plane-projected base. Move the cursor up the hill's slope and watch
//! the cell coords track the visible surface; before PR 2 the
//! ray-vs-Z=0-plane fallback reported the cell at the hill's base.

use std::collections::HashMap;
use std::time::Duration;

use currawong::glam::{Vec2, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, CommandQueue, EngineCtx, Liquid, LiquidId, OrbitRig, Renderer,
    Simulation, SlopeMesher, SquareGrid, TerrainMaterial, TerrainMaterialInstance, TerrainPicker,
    TerrainRenderer, TileCoord, View, ViewConfig, ViewEnvironment, Zone, ZoneId, Zones, wgpu,
    winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const WATER: LiquidId = LiquidId(1);
const BASE_TITLE: &str = "currawong — slope terrain demo";
const TILE_SIZE: f32 = 1.0;

// --- Simulation ----------------------------------------------------------

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

        // 16×16 base patch at h=0.
        for ty in -8..8 {
            for tx in -8..8 {
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
            }
        }

        // Broad rolling hill centred at (2, 2): heights step down from 4 at
        // the peak through 3, 2, 1 in concentric rings. Each step is one
        // tile wide, so the slope mesher produces a 1-unit slope per tile
        // — the canonical "ramped terrain" shape.
        for ty in -8..8 {
            for tx in -8..8 {
                let dx = tx - 2;
                let dy = ty - 2;
                let d2 = dx * dx + dy * dy;
                let h = if d2 == 0 {
                    4
                } else if d2 <= 2 {
                    3
                } else if d2 <= 8 {
                    2
                } else if d2 <= 18 {
                    1
                } else {
                    0
                };
                if h > 0 {
                    terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = h;
                }
            }
        }

        // Small pond on the opposite side: a 2×2 pit at h=-4 filled with
        // 4 steps of water so the surface sits flush with the surrounding
        // ground at z=0.
        for ty in -6..-4 {
            for tx in -6..-4 {
                let tile = terrain.tile_mut(TileCoord::new(tx, ty));
                tile.floor_height = -4;
                tile.liquid = Some(Liquid {
                    kind: WATER,
                    depth: 4,
                });
            }
        }

        Self { zones, main_zone }
    }
}

impl Simulation for Game {
    type Command = ();
    fn tick(&mut self, _: Duration) {}
}

// --- View ----------------------------------------------------------------

struct TerrainView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,
    picker: TerrainPicker<SquareGrid>,
    last_title_hover: Option<TileCoord>,
    material: TerrainMaterial,
    solid_instance: TerrainMaterialInstance,
    liquid_instances: HashMap<LiquidId, TerrainMaterialInstance>,
    terrain: TerrainRenderer,
}

impl View for TerrainView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: BASE_TITLE,
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera {
            far: 200.0,
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(&renderer.device);

        // Frame the terrain: focus on the centre at the height the old
        // auto-orbit pointed at, ~24 units out at ~32° elevation — a
        // close match to the old `radius=20, height=14` shot. The
        // farther bound is bumped above the default 80 to keep the
        // whole 16×16 patch reachable when zoomed out.
        let mut rig = OrbitRig::new(Vec3::new(0.0, 0.0, 0.0));
        rig.pitch = 32.0_f32.to_radians();
        rig.distance = 24.0;
        rig.config.distance_max = 120.0;

        let picker = TerrainPicker::new(SquareGrid, TILE_SIZE);

        let material = TerrainMaterial::new(renderer, camera_binding.layout());

        let solid_instance = material.create_instance(renderer, Vec4::new(1.0, 1.0, 1.0, 1.0));

        let mut liquid_instances = HashMap::new();
        liquid_instances.insert(
            WATER,
            material.create_instance(renderer, Vec4::new(0.25, 0.5, 0.85, 0.55)),
        );

        Self {
            camera,
            camera_binding,
            rig,
            picker,
            last_title_hover: None,
            material,
            solid_instance,
            liquid_instances,
            terrain: TerrainRenderer::new(),
        }
    }

    fn update(&mut self, _: &Game, _: &mut EngineCtx, _: &mut CommandQueue<()>, dt: Duration) {
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

        let zone = sim.zones.get(sim.main_zone).expect("main zone");
        if self.terrain.is_empty() {
            // 45° slopes per height step (`height_unit == tile_size`). The
            // sun-facing and shadow-facing facets of the same cell end up
            // far enough apart in `dot(n, l)` to read as distinct facets
            // under the orbit camera.
            let mesher = SlopeMesher {
                height_unit: 1.0,
                ..SlopeMesher::new()
            };
            self.terrain.rebuild_all(renderer, zone.terrain(), &mesher);
        }

        // Re-pick every frame — same as the flat-terrain demo, but here the
        // GPU ID-buffer path is what actually makes hover correct over the
        // sloped hill. The ray-vs-plane fallback reports the cell at z=0
        // under the cursor; on a hill that's the cell at the *base*, not
        // the slope facet the cursor is visually over. After PR 2 of #56
        // the engine writes per-pixel hit IDs from the terrain shader and
        // we read them back here.
        let viewport = Vec2::new(size.width as f32, size.height as f32);
        self.picker.update(&self.camera, viewport);
        self.picker.set_id_hover(renderer.hit_id_hover());
        let hover_coord = self
            .picker
            .hover()
            .map(|h| TileCoord::new(h.cell.x, h.cell.y));
        if hover_coord != self.last_title_hover {
            let title = match hover_coord {
                Some(c) => format!("{BASE_TITLE} — hover ({}, {})", c.x, c.y),
                None => BASE_TITLE.to_string(),
            };
            renderer.window.set_title(&title);
            self.last_title_hover = hover_coord;
        }

        self.camera_binding.write(&renderer.queue, &self.camera);

        pass.set_pipeline(self.material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.main_zone, &self.solid_instance);

        pass.set_pipeline(self.material.transparent_pipeline());
        self.terrain.draw_liquids(pass, &self.liquid_instances);
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        // Required for `extract_environment` to be called — the engine
        // writes a neutral environment whenever `active_zone` is `None`,
        // which gives full ambient and no directional sun.
        Some(sim.main_zone)
    }

    fn extract_environment(&self, _: &Game, _: ZoneId) -> ViewEnvironment {
        // Slopes rely *entirely* on shading variation to read (no cliff
        // walls to provide color contrast like the flat-tops demos do), so
        // the sun is dimmer here — bright enough to drive the dot-product
        // shading, dim enough that lit slope facets stay below the
        // sRGB-target clipping threshold. With the flat-tops examples'
        // `* 2.2` brightness, every sun-facing surface clamps to 1.0 and
        // the slope variation becomes invisible.
        ViewEnvironment {
            sun_direction: Vec3::new(0.45, 0.35, 0.8).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85) * 1.2,
            ambient: Vec3::new(0.18, 0.20, 0.24),
            sky_color: Vec3::new(0.45, 0.65, 0.95),
            ..ViewEnvironment::neutral()
        }
    }

    fn input(
        &mut self,
        _: &Game,
        ctx: &mut EngineCtx,
        _: &mut CommandQueue<()>,
        event: &WindowEvent,
    ) {
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
            KeyCode::Digit0 => ctx.clock.set_speed(0.0),
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            KeyCode::Digit2 => ctx.clock.set_speed(2.0),
            KeyCode::Digit3 => ctx.clock.set_speed(3.0),
            KeyCode::KeyY => self.rig.config.invert_pitch = !self.rig.config.invert_pitch,
            KeyCode::KeyX => self.rig.config.invert_yaw = !self.rig.config.invert_yaw,
            _ => {}
        }
    }
}

fn main() {
    currawong::run::<TerrainView>(Game::new());
}
