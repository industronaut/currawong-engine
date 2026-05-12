//! Two coordinate-isolated zones connected by a stair tile. Validates the
//! "movement between zones is a storage operation" invariant: stepping onto a
//! stair tile triggers `Zone::remove` + `Zone::insert` (plus a manual
//! transfer of the player's components), the View's `active_zone` follows the
//! player, and a per-zone `extract_environment` makes the swap visible.
//!
//! Layout:
//! - **ground** zone: 16×16 grass field, a water pond in the SW corner, a
//!   raised stone tile at `(4, 4)` marking the stair up.
//! - **upper** zone: 16×16 stone plateau at floor-height 5, a dropped stone
//!   tile at `(-3, -3)` marking the stair down. Different stair coordinates
//!   between the two zones underline that each zone has its own frame.
//!
//! Controls:
//! - WASD / arrows — move the player one tile per key press.
//! - R — reset to the ground starting tile.
//! - 0 / 1 — pause / 1× sim speed (the sun is the only thing actively driven
//!   by the clock).
//! - Esc — quit.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, FlatTopsMesher, Liquid, LiquidId, Renderer, SimEnvironment,
    Simulation, TerrainMaterial, TerrainMaterialInstance, TerrainRenderer, TileCoord,
    UnlitColoredAttribs, UnlitColoredInstance, UnlitColoredMaterial, View, ViewConfig,
    ViewEnvironment, WorldObject, WorldObjectRef, Zone, ZoneId, Zones, sun_direction_for, wgpu,
    winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Simulation ----------------------------------------------------------

const WATER: LiquidId = LiquidId(1);
const HEIGHT_UNIT: f32 = 0.1;
const ZONE_RADIUS: i32 = 8;

/// Marker component on the player. Used to assert post-transition that the
/// player's components landed on the destination zone's registry.
#[derive(Clone, Copy)]
struct Player;

/// Stand-in payload that must follow the player across zones. Survives the
/// transition because we manually move it in [`Game::check_stair_trigger`];
/// `Zone::remove`'s cascade would otherwise drop it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Health(u32);

struct Game {
    zones: Zones,
    ground: ZoneId,
    upper: ZoneId,
    player: WorldObjectRef,
    env: SimEnvironment,
    /// Stair tile in the ground zone — stepping onto it teleports to
    /// `stair_in_upper` in the upper zone.
    stair_in_ground: TileCoord,
    stair_in_upper: TileCoord,
    /// True for the window between a transition firing and the player's
    /// next movement. Without it, the player lands on the partner stair
    /// tile and the next tick immediately bounces them back. Stairs fire
    /// on *entry*, not while standing.
    just_transitioned: bool,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let ground = zones.insert(Zone::new());
        let upper = zones.insert(Zone::new());

        // Different coordinates in each zone — zones are coordinate-isolated.
        let stair_in_ground = TileCoord::new(4, 4);
        let stair_in_upper = TileCoord::new(-3, -3);

        build_ground_zone(zones.get_mut(ground).unwrap(), stair_in_ground);
        build_upper_zone(zones.get_mut(upper).unwrap(), stair_in_upper);

        let start_tile = TileCoord::new(0, 0);
        let zone = zones.get_mut(ground).unwrap();
        let player_id = zone.insert(WorldObject {
            position: tile_to_pos(start_tile, 0),
            rotation: Quat::IDENTITY,
        });
        zone.components_mut().insert(player_id, Player);
        zone.components_mut().insert(player_id, Health(100));
        let player = WorldObjectRef {
            zone: ground,
            id: player_id,
        };

        let mut env = SimEnvironment::new();
        // Short day so the sun visibly moves during a play session.
        env.seconds_per_day = 60.0;
        env.time_of_day = 0.35;

        Self {
            zones,
            ground,
            upper,
            player,
            env,
            stair_in_ground,
            stair_in_upper,
            just_transitioned: false,
        }
    }

    fn current_player_tile(&self) -> TileCoord {
        let p = self.player.resolve(&self.zones).expect("player exists");
        TileCoord::new(p.position.x.floor() as i32, p.position.y.floor() as i32)
    }

    /// Move the player one tile within their current zone, clamped to the
    /// 16×16 area each zone explicitly laid down.
    fn step_player(&mut self, dx: i32, dy: i32) {
        let cur = self.current_player_tile();
        let next = TileCoord::new(cur.x + dx, cur.y + dy);
        if !in_bounds(next) {
            return;
        }
        let zone = self.zones.get_mut(self.player.zone).unwrap();
        let floor = zone.terrain().tile_or_default(next).floor_height;
        zone.get_mut(self.player.id).unwrap().position = tile_to_pos(next, floor);
        // The player just *entered* a tile. Re-arm the stair trigger so
        // walking back onto a stair will fire it again.
        self.just_transitioned = false;
    }

    /// Drop the player wherever they are and recreate them at the ground
    /// starting tile. Exercises `Zone::remove`'s component cascade alongside
    /// the explicit transfer path used by transitions.
    fn reset_player(&mut self) {
        let src = self.zones.get_mut(self.player.zone).unwrap();
        src.remove(self.player.id);

        let zone = self.zones.get_mut(self.ground).unwrap();
        let id = zone.insert(WorldObject {
            position: tile_to_pos(TileCoord::new(0, 0), 0),
            rotation: Quat::IDENTITY,
        });
        zone.components_mut().insert(id, Player);
        zone.components_mut().insert(id, Health(100));
        self.player = WorldObjectRef {
            zone: self.ground,
            id,
        };
        self.just_transitioned = false;
    }

    /// If the player stands on the active zone's stair tile, move them into
    /// the partner zone at the partner's stair tile. Components ride along.
    fn check_stair_trigger(&mut self) {
        // Suppress until the player has stepped off the partner stair —
        // otherwise the player bounces between zones every tick.
        if self.just_transitioned {
            return;
        }
        let cur = self.current_player_tile();
        let (this_stair, dest_zone, dest_stair) = if self.player.zone == self.ground {
            (self.stair_in_ground, self.upper, self.stair_in_upper)
        } else {
            (self.stair_in_upper, self.ground, self.stair_in_ground)
        };
        if cur != this_stair {
            return;
        }

        // Snapshot components from the source zone. After this, the source's
        // Components registry holds no entries for this id. We could also
        // rely on `Zone::remove`'s cascade — taking them out by hand here is
        // what lets them ride across the boundary.
        let src = self.zones.get_mut(self.player.zone).unwrap();
        let player_marker = src.components_mut().remove::<Player>(self.player.id);
        let health = src.components_mut().remove::<Health>(self.player.id);
        let mut obj = src.remove(self.player.id).expect("player object");

        // Translate the position into the destination zone's local frame.
        let dest_floor = self
            .zones
            .get(dest_zone)
            .unwrap()
            .terrain()
            .tile_or_default(dest_stair)
            .floor_height;
        obj.position = tile_to_pos(dest_stair, dest_floor);

        let dest = self.zones.get_mut(dest_zone).unwrap();
        let new_id = dest.insert(obj);
        if let Some(p) = player_marker {
            dest.components_mut().insert(new_id, p);
        }
        if let Some(h) = health {
            dest.components_mut().insert(new_id, h);
        }

        // Runtime assertion of the load-bearing invariant: components must
        // land on the destination zone and be absent on the source. Cheap
        // enough to keep on in release; the issue's acceptance criterion is
        // exactly this check.
        debug_assert!(
            self.zones
                .get(dest_zone)
                .unwrap()
                .components()
                .get::<Player>(new_id)
                .is_some(),
            "player marker missing on dest zone after transition"
        );
        debug_assert!(
            self.zones
                .get(self.player.zone)
                .unwrap()
                .components()
                .get::<Player>(self.player.id)
                .is_none(),
            "player marker still on source zone after transition"
        );

        self.player = WorldObjectRef {
            zone: dest_zone,
            id: new_id,
        };
        self.just_transitioned = true;
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        self.env.advance(dt.as_secs_f32());
        self.check_stair_trigger();
    }
}

fn tile_to_pos(tile: TileCoord, floor_height: i32) -> Vec3 {
    // Match FlatTopsMesher's tile-centre convention with our height_unit.
    Vec3::new(
        tile.x as f32 + 0.5,
        tile.y as f32 + 0.5,
        floor_height as f32 * HEIGHT_UNIT,
    )
}

fn in_bounds(tile: TileCoord) -> bool {
    tile.x >= -ZONE_RADIUS && tile.x < ZONE_RADIUS && tile.y >= -ZONE_RADIUS && tile.y < ZONE_RADIUS
}

fn build_ground_zone(zone: &mut Zone, stair: TileCoord) {
    let terrain = zone.terrain_mut();
    // 16×16 grass field, flat at h=0.
    for ty in -ZONE_RADIUS..ZONE_RADIUS {
        for tx in -ZONE_RADIUS..ZONE_RADIUS {
            terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
        }
    }
    // Water pond — a 3×3 pit in the SW corner.
    for ty in -7..-4 {
        for tx in -7..-4 {
            let tile = terrain.tile_mut(TileCoord::new(tx, ty));
            tile.floor_height = -8;
            tile.liquid = Some(Liquid {
                kind: WATER,
                depth: 8,
            });
        }
    }
    // Raised stair marker (h=2) so the destination is visually obvious.
    terrain.tile_mut(stair).floor_height = 2;
}

fn build_upper_zone(zone: &mut Zone, stair: TileCoord) {
    let terrain = zone.terrain_mut();
    // 16×16 plateau at h=5 — distinct elevation makes the zone switch obvious.
    for ty in -ZONE_RADIUS..ZONE_RADIUS {
        for tx in -ZONE_RADIUS..ZONE_RADIUS {
            terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 5;
        }
    }
    // A small decorative bump near the centre.
    for ty in -1..2 {
        for tx in -1..2 {
            terrain.tile_mut(TileCoord::new(tx, ty)).floor_height = 6;
        }
    }
    // Stair-down marker: drop the stair tile a couple of steps.
    terrain.tile_mut(stair).floor_height = 3;
}

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct MultiZoneView {
    camera: Camera,
    camera_binding: CameraBinding,
    terrain_material: TerrainMaterial,
    ground_tint: TerrainMaterialInstance,
    upper_tint: TerrainMaterialInstance,
    liquid_instances: HashMap<LiquidId, TerrainMaterialInstance>,
    terrain_cache: TerrainRenderer,
    /// Which zone `terrain_cache` is currently meshed for; `None` until the
    /// first frame. When the active zone changes we drop the cached chunks
    /// and rebuild from the new zone — that's the "view state is recoverable
    /// from sim state" invariant in practice.
    cached_zone: Option<ZoneId>,
    player_material: UnlitColoredMaterial,
    player_instance: UnlitColoredInstance,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    player_attribs: wgpu::Buffer,
    cube_index_count: u32,
    started: Instant,
}

#[rustfmt::skip]
const CUBE_POSITIONS: &[[f32; 3]] = &[
    [-0.30, -0.30, 0.00],
    [ 0.30, -0.30, 0.00],
    [ 0.30,  0.30, 0.00],
    [-0.30,  0.30, 0.00],
    [-0.30, -0.30, 0.60],
    [ 0.30, -0.30, 0.60],
    [ 0.30,  0.30, 0.60],
    [-0.30,  0.30, 0.60],
];

#[rustfmt::skip]
const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // -Z
    4, 6, 5, 4, 7, 6, // +Z
    0, 3, 7, 0, 7, 4, // -X
    1, 5, 6, 1, 6, 2, // +X
    3, 2, 6, 3, 6, 7, // +Y
    0, 4, 5, 0, 5, 1, // -Y
];

impl View for MultiZoneView {
    type Sim = Game;

    fn init(renderer: &Renderer) -> (Self, ViewConfig) {
        use wgpu::util::DeviceExt;
        let device = &renderer.device;

        let camera = Camera::default();
        let camera_binding = CameraBinding::new(device);
        let terrain_material = TerrainMaterial::new(renderer, camera_binding.layout());
        let ground_tint = terrain_material.create_instance(renderer, Vec4::new(1.0, 1.0, 1.0, 1.0));
        let upper_tint =
            terrain_material.create_instance(renderer, Vec4::new(0.95, 0.95, 1.0, 1.0));
        let mut liquid_instances = HashMap::new();
        liquid_instances.insert(
            WATER,
            terrain_material.create_instance(renderer, Vec4::new(0.25, 0.5, 0.85, 0.55)),
        );

        let player_material = UnlitColoredMaterial::new(renderer, camera_binding.layout());
        let player_instance =
            player_material.create_instance(renderer, Vec4::new(0.95, 0.4, 0.2, 1.0));

        let cube_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("player cube vertices"),
            contents: bytemuck::cast_slice(CUBE_POSITIONS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("player cube indices"),
            contents: bytemuck::cast_slice(CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let player_attribs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("player attribs"),
            size: std::mem::size_of::<UnlitColoredAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        (
            Self {
                camera,
                camera_binding,
                terrain_material,
                ground_tint,
                upper_tint,
                liquid_instances,
                terrain_cache: TerrainRenderer::new(),
                cached_zone: None,
                player_material,
                player_instance,
                cube_vertices,
                cube_indices,
                player_attribs,
                cube_index_count: CUBE_INDICES.len() as u32,
                started: Instant::now(),
            },
            ViewConfig {
                title: "currawong — multi-zone (active: ground)",
                depth_format: Some(DEPTH_FORMAT),
                ..Default::default()
            },
        )
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.player.zone)
    }

    fn extract_environment(&self, sim: &Game, zone: ZoneId) -> ViewEnvironment {
        // Same `time_of_day` for both zones — only the *appearance* differs,
        // proving the per-zone branch happens in the view, not the sim.
        let sun = sun_direction_for(sim.env.time_of_day);
        if zone == sim.ground {
            ViewEnvironment {
                sun_direction: sun,
                sun_color: Vec3::new(1.0, 0.95, 0.85) * 2.4,
                ambient: Vec3::new(0.35, 0.40, 0.45),
                sky_color: Vec3::new(0.45, 0.65, 0.95),
            }
        } else {
            ViewEnvironment {
                sun_direction: sun,
                sun_color: Vec3::new(0.80, 0.85, 1.0) * 1.2,
                ambient: Vec3::new(0.18, 0.20, 0.26),
                sky_color: Vec3::new(0.18, 0.22, 0.30),
            }
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

        // Tear the terrain cache down whenever the active zone changes. The
        // next branch rebuilds it from the new zone's sim state.
        let active = sim.player.zone;
        if self.cached_zone != Some(active) {
            self.terrain_cache = TerrainRenderer::new();
            self.cached_zone = Some(active);
            // Surface the swap in the window title — handy when running the
            // example without the egui feature.
            let label = if active == sim.ground {
                "ground"
            } else {
                "upper"
            };
            renderer
                .window
                .set_title(&format!("currawong — multi-zone (active: {label})"));
        }
        self.camera.zone = Some(active);

        let zone = sim.zones.get(active).expect("active zone");
        if self.terrain_cache.is_empty() {
            // Two meshers — different top/wall palettes so the zones read as
            // distinct surfaces.
            let mesher = if active == sim.ground {
                FlatTopsMesher {
                    height_unit: HEIGHT_UNIT,
                    top_color: [0.42, 0.62, 0.30, 1.0],
                    wall_color: [0.35, 0.30, 0.20, 1.0],
                    ..FlatTopsMesher::new()
                }
            } else {
                FlatTopsMesher {
                    height_unit: HEIGHT_UNIT,
                    top_color: [0.55, 0.55, 0.60, 1.0],
                    wall_color: [0.30, 0.30, 0.35, 1.0],
                    ..FlatTopsMesher::new()
                }
            };
            self.terrain_cache
                .rebuild_all(renderer, zone.terrain(), &mesher);
        }
        let terrain_tint = if active == sim.ground {
            &self.ground_tint
        } else {
            &self.upper_tint
        };

        let player = sim
            .player
            .resolve(&sim.zones)
            .expect("player resolves while alive");

        // Wall-clock orbit so the camera keeps moving even at sim speed 0 —
        // makes sim/view decoupling visible.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 9.0;
        let angle = t * 0.20;
        self.camera.position =
            player.position + Vec3::new(angle.sin() * radius, angle.cos() * radius, 6.0);
        self.camera.target = player.position + Vec3::new(0.0, 0.0, 0.4);
        self.camera.far = 200.0;
        self.camera_binding.write(&renderer.queue, &self.camera);

        // --- Terrain pass ------------------------------------------------
        pass.set_pipeline(self.terrain_material.opaque_pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        self.terrain_cache.draw_solid(pass, terrain_tint);
        pass.set_pipeline(self.terrain_material.transparent_pipeline());
        self.terrain_cache
            .draw_liquids(pass, &self.liquid_instances);

        // --- Player pass -------------------------------------------------
        let spin = Quat::from_rotation_z(t * 0.8);
        let model = Mat4::from_rotation_translation(player.rotation * spin, player.position);
        let attribs = UnlitColoredAttribs::new(model, Vec4::ONE);
        renderer
            .queue
            .write_buffer(&self.player_attribs, 0, bytemuck::bytes_of(&attribs));
        pass.set_pipeline(self.player_material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, self.player_instance.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.cube_vertices.slice(..));
        pass.set_vertex_buffer(1, self.player_attribs.slice(..));
        pass.set_index_buffer(self.cube_indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.cube_index_count, 0, 0..1);
    }

    fn input(&mut self, sim: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
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
            KeyCode::KeyW | KeyCode::ArrowUp => sim.step_player(0, 1),
            KeyCode::KeyS | KeyCode::ArrowDown => sim.step_player(0, -1),
            KeyCode::KeyA | KeyCode::ArrowLeft => sim.step_player(-1, 0),
            KeyCode::KeyD | KeyCode::ArrowRight => sim.step_player(1, 0),
            KeyCode::KeyR => sim.reset_player(),
            KeyCode::Digit0 => ctx.clock.set_speed(0.0),
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            _ => {}
        }
    }
}

fn main() {
    currawong::run::<MultiZoneView>(Game::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The acceptance criterion: after a transition the player's components
    /// live on the destination zone and the source zone holds nothing for the
    /// old id.
    #[test]
    fn transition_moves_components_across_zones() {
        let mut game = Game::new();
        let start_zone = game.player.zone;
        let start_id = game.player.id;
        assert_eq!(start_zone, game.ground);

        // Walk the player onto the stair tile. `step_player` only moves one
        // tile, so step from (0,0) to (4,4).
        for _ in 0..4 {
            game.step_player(1, 0);
        }
        for _ in 0..4 {
            game.step_player(0, 1);
        }
        assert_eq!(game.current_player_tile(), game.stair_in_ground);

        // The transition happens inside `tick`.
        game.tick(Duration::from_millis(16));

        // After: zone is upper, player tile is the upper stair tile.
        assert_eq!(game.player.zone, game.upper);
        assert_eq!(game.current_player_tile(), game.stair_in_upper);

        // Components present on destination, absent on source.
        let upper = game.zones.get(game.upper).unwrap();
        assert!(upper.components().get::<Player>(game.player.id).is_some());
        assert_eq!(
            upper.components().get::<Health>(game.player.id),
            Some(&Health(100))
        );
        let ground = game.zones.get(game.ground).unwrap();
        assert!(ground.components().get::<Player>(start_id).is_none());
        assert!(ground.components().get::<Health>(start_id).is_none());
        // The old id no longer resolves to any object on the source zone.
        assert!(!ground.contains(start_id));
    }

    /// After landing on the partner stair tile, further ticks must NOT
    /// re-trigger the transition — stairs fire on entry, not on standing.
    /// Regression: an earlier version bounced the player between zones
    /// every tick.
    #[test]
    fn transition_does_not_bounce_on_subsequent_ticks() {
        let mut game = Game::new();
        for _ in 0..4 {
            game.step_player(1, 0);
        }
        for _ in 0..4 {
            game.step_player(0, 1);
        }
        game.tick(Duration::from_millis(16));
        assert_eq!(game.player.zone, game.upper);

        // Sit still for many ticks. The player is standing on the upper
        // stair tile but should not transition until they leave and return.
        for _ in 0..10 {
            game.tick(Duration::from_millis(16));
        }
        assert_eq!(game.player.zone, game.upper);
        assert_eq!(game.current_player_tile(), game.stair_in_upper);
    }

    /// Stepping back onto the upper stair should bounce the player back to
    /// the ground zone. Round-trip exercises both directions.
    #[test]
    fn transition_is_reversible() {
        let mut game = Game::new();
        // First trip: ground → upper.
        for _ in 0..4 {
            game.step_player(1, 0);
        }
        for _ in 0..4 {
            game.step_player(0, 1);
        }
        game.tick(Duration::from_millis(16));
        assert_eq!(game.player.zone, game.upper);

        // We're now standing on the upper stair tile (-3, -3). Step off, then
        // back onto it.
        game.step_player(1, 0);
        assert_eq!(game.player.zone, game.upper, "moving doesn't swap zones");
        game.step_player(-1, 0);
        assert_eq!(game.current_player_tile(), game.stair_in_upper);
        game.tick(Duration::from_millis(16));

        assert_eq!(game.player.zone, game.ground);
        assert_eq!(game.current_player_tile(), game.stair_in_ground);
        let ground = game.zones.get(game.ground).unwrap();
        assert!(ground.components().get::<Player>(game.player.id).is_some());
    }
}
