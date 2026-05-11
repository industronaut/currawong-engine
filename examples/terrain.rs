//! Tile-grid terrain with flat tops, cliff walls, and a pool of water.
//!
//! Builds a small zone with a multi-step hill in the +X +Z corner and a 3×3
//! pool of water in the -X -Z corner. The camera orbits on a wall-clock
//! timer (so the view stays alive even at sim speed 0) while the static
//! terrain re-issues its draw calls each frame.
//!
//! Demonstrates the slice we just landed: [`Zone::terrain`] sim data →
//! [`FlatTopsMesher`] CPU geometry → [`TerrainRenderer`] GPU upload →
//! [`TerrainMaterial`] opaque + transparent pipelines.
//!
//! Controls: 0 pause, 1/2/3 set sim speed (no visible effect — sim is
//! inert), Esc to quit.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, Liquid, LiquidId, Renderer, Simulation,
    TerrainMaterial, TerrainMaterialInstance, TerrainRenderer, TileCoord, View, Zone, ZoneId,
    Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const WATER: LiquidId = LiquidId(1);

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

        // Allocate a 16×16 area of ground centred on the origin. All tiles
        // default to walkable + h=0; we then carve heights and water.
        for ty in -8..8 {
            for tx in -8..8 {
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
            }
        }

        // Stepped hill centred at (4, 4): three height bands.
        for ty in -8..8 {
            for tx in -8..8 {
                let dx = tx - 4;
                let dy = ty - 4;
                let d2 = dx * dx + dy * dy;
                let h = if d2 < 3 {
                    3
                } else if d2 < 8 {
                    2
                } else if d2 < 16 {
                    1
                } else {
                    0
                };
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = h;
            }
        }

        // 3×3 pool of water in the -X -Z corner: floor dropped to -1, full
        // depth water sits the surface back at y=0 (flush with surrounding
        // ground).
        for ty in -6..-3 {
            for tx in -6..-3 {
                let tile = terrain.tile_mut(TileCoord::new(tx, ty));
                tile.floor_height = -1;
                tile.liquid = Some(Liquid {
                    kind: WATER,
                    depth: 255,
                });
            }
        }

        Self { zones, main_zone }
    }
}

impl Simulation for Game {
    fn tick(&mut self, _: Duration) {}
}

// --- View ----------------------------------------------------------------

struct TerrainView {
    camera: Camera,
    camera_binding: CameraBinding,
    material: TerrainMaterial,
    solid_instance: TerrainMaterialInstance,
    liquid_instances: HashMap<LiquidId, TerrainMaterialInstance>,
    terrain: TerrainRenderer,
    started: Instant,
}

impl View for TerrainView {
    type Sim = Game;

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(&renderer.device);
        let material = TerrainMaterial::new(renderer, camera_binding.layout());

        // Solid tint = white so per-vertex top/wall colours show through.
        let solid_instance = material.create_instance(renderer, Vec4::new(1.0, 1.0, 1.0, 1.0));

        // One material instance per liquid kind. Tint colour is the liquid's
        // colour; alpha < 1 enables see-through.
        let mut liquid_instances = HashMap::new();
        liquid_instances.insert(
            WATER,
            material.create_instance(renderer, Vec4::new(0.25, 0.5, 0.85, 0.55)),
        );

        Self {
            camera,
            camera_binding,
            material,
            solid_instance,
            liquid_instances,
            terrain: TerrainRenderer::new(),
            started: Instant::now(),
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

        // Lazy first-frame mesh — init() doesn't have access to the sim, so
        // upload happens here on the first render call. Re-meshing on edit
        // is the caller's job (this demo's terrain is static).
        let zone = sim.zones.get(sim.main_zone).expect("main zone");
        if self.terrain.is_empty() {
            self.terrain
                .rebuild_all(renderer, zone.terrain(), &FlatTopsMesher::new());
        }

        // Wall-clock orbit so pausing the sim doesn't freeze the camera —
        // makes the sim/view decoupling visible.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 18.0;
        let angle = t * 0.25;
        self.camera.position = Vec3::new(angle.sin() * radius, 12.0, angle.cos() * radius);
        self.camera.target = Vec3::new(0.0, 0.5, 0.0);
        self.camera.far = 200.0;
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Opaque solids first so depth is populated before we draw liquids.
        pass.set_pipeline(self.material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        self.terrain.draw_solid(pass, &self.solid_instance);

        // Liquids over the top — alpha-blended, depth-test on but no depth
        // write (set in the material).
        pass.set_pipeline(self.material.transparent_pipeline());
        self.terrain.draw_liquids(pass, &self.liquid_instances);
    }

    fn input(&mut self, _: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
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
            _ => {}
        }
    }

    fn title() -> &'static str {
        "currawong — terrain demo"
    }

    fn depth_format() -> Option<wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }
}

fn main() {
    currawong::run::<TerrainView>(Game::new());
}
