//! Tile-grid terrain with flat tops, cliff walls, and a pool of water —
//! plus a hover highlight driven by mouse-over picking.
//!
//! Builds a small zone with a multi-step hill in the +X +Y corner and a 3×3
//! pool of water in the -X -Y corner. An [`OrbitRig`] drives the camera so
//! the user can park it over a region of interest; a [`TerrainPicker`]
//! converts the cursor into a tile coordinate each frame; a [`CellHighlight`]
//! paints a translucent yellow overlay over the picked tile.
//!
//! Demonstrates the picking slice on top of the existing terrain pipeline:
//! ray-cast picker → tile coord → highlight draw, with the hovered cell
//! also echoed into the window title.
//!
//! Controls:
//! - Right-click drag — rotate the camera around the focal point.
//! - W / A / S / D — pan the focal point on the ground.
//! - Scroll wheel — zoom.
//! - 0 pause sim, 1/2/3 set sim speed (no visible effect — sim is inert).
//! - Mouse over the terrain — picked tile gets a translucent yellow
//!   overlay; the tile coordinate also shows in the title bar.
//! - Esc to quit.

use std::collections::HashMap;
use std::time::Duration;

use currawong::glam::{Vec2, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, CellHighlight, EngineCtx, FlatTopsMesher, Liquid, LiquidId, OrbitRig,
    Renderer, Simulation, SquareGrid, TerrainMaterial, TerrainMaterialInstance, TerrainPicker,
    TerrainRenderer, TileCoord, View, ViewConfig, ViewEnvironment, Zone, ZoneId, Zones, wgpu,
    winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const WATER: LiquidId = LiquidId(1);
const TILE_SIZE: f32 = 1.0;
const HEIGHT_UNIT: f32 = 0.1;
const BASE_TITLE: &str = "currawong — terrain demo";

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
                    tx
                } else if d2 < 8 {
                    3
                } else if d2 < 16 {
                    1
                } else {
                    0
                };
                terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = h;
            }
        }

        // 3×3 pool of water in the -X -Y corner: floor dropped to -10 (a
        // deep pit), filled with 10 steps of water so the surface sits flush
        // with the surrounding ground at z=0.
        for ty in -6..-3 {
            for tx in -6..-3 {
                let tile = terrain.tile_mut(TileCoord::new(tx, ty));
                tile.floor_height = -10;
                tile.liquid = Some(Liquid {
                    kind: WATER,
                    depth: 10,
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
    rig: OrbitRig,
    picker: TerrainPicker<SquareGrid>,
    highlight: CellHighlight,
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
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(&renderer.device);
        // Start the rig parked over the origin, far enough out to see the
        // hill, the pool, and a generous border of flat ground.
        let mut rig = OrbitRig::new(Vec3::new(0.0, 0.0, 0.0));
        rig.distance = 18.0;
        rig.pitch = 55.0_f32.to_radians();
        let picker = TerrainPicker::new(SquareGrid, TILE_SIZE);
        // Warm yellow at ~50% alpha so the tile colour shows through the
        // overlay. The highlight starts empty (no cell set) so the first
        // frame draws nothing — `set_cell` populates it once the cursor
        // lands on terrain.
        let highlight = CellHighlight::new(
            renderer,
            camera_binding.layout(),
            Vec4::new(1.0, 0.85, 0.2, 0.5),
        );
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
            rig,
            picker,
            highlight,
            last_title_hover: None,
            material,
            solid_instance,
            liquid_instances,
            terrain: TerrainRenderer::new(),
        }
    }

    fn update(&mut self, _: &Game, _: &mut EngineCtx, dt: Duration) {
        // Rig integrates held-WASD with wall-clock dt so panning keeps
        // working at sim speed 0; same pattern as the camera demo.
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

        // Lazy first-frame mesh — init() doesn't have access to the sim, so
        // upload happens here on the first render call. Re-meshing on edit
        // is the caller's job (this demo's terrain is static).
        let zone = sim.zones.get(sim.main_zone).expect("main zone");
        if self.terrain.is_empty() {
            // Short height steps so a 1-level cliff is one-tenth of a tile
            // tall rather than a full cube — closer to the RimWorld / DF
            // sim-game aesthetic than a Minecraft voxel look.
            let mesher = FlatTopsMesher {
                tile_size: TILE_SIZE,
                height_unit: HEIGHT_UNIT,
                ..FlatTopsMesher::new()
            };
            self.terrain.rebuild_all(renderer, zone.terrain(), &mesher);
        }

        // Re-pick every frame — the cursor may not have moved but the
        // camera might have, which changes which cell the same pixel
        // covers. Run after the camera's aspect ratio is up to date so the
        // unproject uses the right projection matrix. Pull the GPU
        // ID-buffer result in too; on sloped terrain it disagrees with the
        // ray-plane fallback and takes precedence inside `hover()`.
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

        // Drive the fill overlay off the picker. The highlight Z sits a
        // hair above the tile top so the fill doesn't Z-fight the mesh
        // beneath it. With #56 PR 2 the picker prefers the GPU ID-buffer
        // result over its ray-vs-plane fallback, so the cell coordinate is
        // correct even on slopes; the overlay's Z still reads `floor_height`
        // directly from the sim so the outline tracks the visible mesh top.
        match hover_coord {
            Some(coord) => {
                let h = zone.terrain().tile_or_default(coord).floor_height;
                let z = h as f32 * HEIGHT_UNIT + 0.02;
                self.highlight
                    .set_cell(renderer, &SquareGrid, coord, z, TILE_SIZE);
            }
            None => self.highlight.clear(),
        }

        self.camera_binding.write(&renderer.queue, &self.camera);

        // Opaque solids first so depth is populated before we draw liquids.
        pass.set_pipeline(self.material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        self.terrain
            .draw_solid(pass, renderer, sim.main_zone, &self.solid_instance);

        // Liquids over the top — alpha-blended, depth-test on but no depth
        // write (set in the material).
        pass.set_pipeline(self.material.transparent_pipeline());
        self.terrain.draw_liquids(pass, &self.liquid_instances);

        // Hover outline last so it lands on top of solids and liquids.
        // No-op when nothing is hovered.
        self.highlight.draw(pass);
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.main_zone)
    }

    fn extract_environment(&self, _: &Game, _: ZoneId) -> ViewEnvironment {
        // Fixed afternoon sun — high enough to light tops well, oblique
        // enough to shade walls visibly. Brighter sun + low ambient so
        // cliff faces read by shading, not flat colour.
        ViewEnvironment {
            sun_direction: Vec3::new(0.45, 0.35, 0.8).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85) * 2.2,
            ambient: Vec3::new(0.30, 0.32, 0.38),
            sky_color: Vec3::new(0.45, 0.65, 0.95),
        }
    }

    fn input(&mut self, _: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
        // Camera rig and picker both see every event. The picker filters to
        // CursorMoved / CursorLeft; the rig handles MB / scroll / WASD.
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
            _ => {}
        }
    }
}

fn main() {
    currawong::run::<TerrainView>(Game::new());
}
