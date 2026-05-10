//! Headless simulation tick — no window, no GPU, no rendering.
//!
//! Demonstrates that the simulation runs independently of any view. To prove
//! at the build level that the sim layer does not depend on `wgpu` or `winit`,
//! run with the render feature disabled:
//!
//! ```text
//! cargo run --example headless --no-default-features
//! ```

use std::time::{Duration, Instant};

use currawong::glam::{Quat, Vec3};
use currawong::{Simulation, WorldObject, WorldObjectRef, Zone, Zones};

struct Game {
    zones: Zones,
    obj_a: WorldObjectRef,
    obj_b: WorldObjectRef,
    elapsed: Duration,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");

        let obj_a_id = zone.insert(WorldObject {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        });
        let obj_b_id = zone.insert(WorldObject {
            position: Vec3::new(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        });

        Self {
            zones,
            obj_a: WorldObjectRef { zone: zone_id, id: obj_a_id },
            obj_b: WorldObjectRef { zone: zone_id, id: obj_b_id },
            elapsed: Duration::ZERO,
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        self.elapsed += dt;
        let dx = dt.as_secs_f32();
        for (_, zone) in self.zones.iter_mut() {
            for (_, obj) in zone.iter_mut() {
                obj.position.x += dx;
            }
        }
    }
}

fn main() {
    let mut game = Game::new();
    let target_dt = Duration::from_millis(16);
    let mut last = Instant::now();

    println!("running headless tick for 100 frames at ~60Hz");
    for frame in 0..100 {
        let now = Instant::now();
        let dt = now - last;
        last = now;
        game.tick(dt);
        if frame % 20 == 0 {
            println!(
                "frame {frame:3}: elapsed={:.3}s obj_a.x={:.3} obj_b.x={:.3}",
                game.elapsed.as_secs_f32(),
                game.obj_a.resolve(&game.zones).unwrap().position.x,
                game.obj_b.resolve(&game.zones).unwrap().position.x,
            );
        }
        if let Some(remaining) = target_dt.checked_sub(now.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
    let zone_count = game.zones.len();
    let obj_count: usize = game.zones.iter().map(|(_, z)| z.len()).sum();
    println!(
        "done. total elapsed: {:.3?}, zones: {zone_count}, objects: {obj_count}",
        game.elapsed,
    );
}
