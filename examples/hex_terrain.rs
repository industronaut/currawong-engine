//! Flat-top hex terrain. Mirrors the square `terrain` example but builds the
//! zone over [`HexGrid`] instead — proves the same `FlatTopsMesher` instance
//! and `TerrainRenderer` work unchanged for both grid topologies.
//!
//! Layout: a small hex-shaped patch of cells around the axial origin, with a
//! three-step hill biased to one side and a 3-cell pool of water on the
//! opposite side. The mesher renders each cell as a flat hexagonal top with
//! wall quads dropping to lower neighbours.
//!
//! Controls: 0 pause, 1/2/3 set sim speed, Esc to quit.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{IVec2, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, HexGrid, Liquid, LiquidId, Renderer,
    Simulation, TerrainMaterial, TerrainMaterialInstance, TerrainRenderer, View, ViewConfig, Zone,
    ZoneId, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const WATER: LiquidId = LiquidId(1);

// --- Simulation ----------------------------------------------------------

struct Game {
    zones: Zones<HexGrid>,
    main_zone: ZoneId,
}

impl Game {
    fn new() -> Self {
        let mut zones: Zones<HexGrid> = Zones::new();
        let main_zone = zones.insert(Zone::with_grid(HexGrid));
        let zone = zones.get_mut(main_zone).expect("just inserted");
        let terrain = zone.terrain_mut();

        // Allocate a hex-shaped patch of radius 5 around the axial origin.
        // For axial coords (q, r), the hex disc of radius N is
        // `|q| ≤ N ∧ |r| ≤ N ∧ |q + r| ≤ N`.
        let radius: i32 = 5;
        for q in -radius..=radius {
            for r in -radius..=radius {
                if q.abs() <= radius && r.abs() <= radius && (q + r).abs() <= radius {
                    terrain.tile_mut(IVec2::new(q, r)).floor_height = 0;
                }
            }
        }

        // Three-step hill biased to the (+q, +r) side. Distance is measured
        // in axial steps via the cube-coord max-of-three formula.
        let centre = IVec2::new(2, 2);
        for q in -radius..=radius {
            for r in -radius..=radius {
                if q.abs() > radius || r.abs() > radius || (q + r).abs() > radius {
                    continue;
                }
                let d = axial_distance(IVec2::new(q, r), centre);
                let h = match d {
                    0 => 3,
                    1 => 2,
                    2 => 1,
                    _ => 0,
                };
                if h > 0 {
                    terrain.tile_mut(IVec2::new(q, r)).floor_height = h;
                }
            }
        }

        // Pool of water on the opposite (-q, -r) side: a 3-cell cluster
        // dropped to h=-10 and filled with 10 steps of water so the surface
        // sits flush with the surrounding ground at z=0.
        for (q, r) in [(-3, -1), (-3, 0), (-2, -1)] {
            let tile = terrain.tile_mut(IVec2::new(q, r));
            tile.floor_height = -10;
            tile.liquid = Some(Liquid {
                kind: WATER,
                depth: 10,
            });
        }

        Self { zones, main_zone }
    }
}

/// Axial-coord distance between two hexes. Equivalent to the cube-coord
/// Chebyshev distance: `(|dq| + |dr| + |dq + dr|) / 2`.
fn axial_distance(a: IVec2, b: IVec2) -> i32 {
    let d = a - b;
    (d.x.abs() + d.y.abs() + (d.x + d.y).abs()) / 2
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

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — hex terrain demo",
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(&renderer.device);
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

        let zone = sim.zones.get(sim.main_zone).expect("main zone");
        if self.terrain.is_empty() {
            // Same mesher as the square demo — only the grid changes.
            let mesher = FlatTopsMesher {
                height_unit: 0.1,
                ..FlatTopsMesher::new()
            };
            self.terrain.rebuild_all(renderer, zone.terrain(), &mesher);
        }

        // Wall-clock orbit so pausing the sim doesn't freeze the camera.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 18.0;
        let angle = t * 0.25;
        self.camera.position = Vec3::new(angle.sin() * radius, angle.cos() * radius, 12.0);
        self.camera.target = Vec3::new(0.0, 0.0, 0.5);
        self.camera.far = 200.0;
        self.camera_binding.write(&renderer.queue, &self.camera);

        pass.set_pipeline(self.material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        self.terrain.draw_solid(pass, &self.solid_instance);

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
}

fn main() {
    currawong::run::<TerrainView>(Game::new());
}
