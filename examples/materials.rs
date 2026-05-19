//! Material template/instance/per-instance-attribs split, demonstrated.
//!
//! Six static cubes, half tagged `Red`, half `Gold`. The View holds one
//! [`UnlitColoredMaterial`] (template — pipeline + bgl) and two instances
//! ([`MaterialInstanceId::Red`], [`MaterialInstanceId::Gold`]). Each frame the
//! extract step walks the sim and pushes per-cube
//! [`MeshInstanceAttribs`] (model matrix + tint) into the bucket keyed by
//! the cube's material id. One instanced draw per material instance.
//!
//! The three tiers in one place:
//! - template — `UnlitColoredMaterial` (one)
//! - instance — `UnlitColoredInstance` x 2 (red, gold)
//! - per-instance attribs — `MeshInstanceAttribs` packed into instance buffers
//!
//! Controls: Esc to quit.

use std::time::Instant;

use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, CommandQueue, EngineCtx, Facing, InstanceBuckets,
    MaterialInstanceRegistry, MeshInstanceAttribs, Renderer, SimPos, Simulation,
    UnlitColoredInstance, UnlitColoredMaterial, View, ViewConfig, WorldTransform, Zone, Zones,
    wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ----------------------------------------------------------------

/// Identifies which material instance a sim object renders with. Sim-side
/// tag — the View decides what each id means visually.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MaterialInstanceId {
    Red,
    Gold,
}

/// Per-object tint multiplier. Shows that per-instance attribs vary
/// independently of the material-level base colour.
struct Tint(Vec4);

struct Game {
    zones: Zones,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zid = zones.insert(Zone::new());
        let zone = zones.get_mut(zid).expect("just inserted");

        // Three reds on the left, three golds on the right. Vary Z per row
        // and tint per cube to show per-instance attribs are independent.
        let reds = [
            (Vec3::new(-2.2, 0.0, 0.8), Vec4::new(1.0, 1.0, 1.0, 1.0)),
            (Vec3::new(-2.2, 0.0, 0.0), Vec4::new(0.7, 0.7, 0.7, 1.0)),
            (Vec3::new(-2.2, 0.0, -0.8), Vec4::new(0.4, 0.4, 0.4, 1.0)),
        ];
        let golds = [
            (Vec3::new(2.2, 0.0, 0.8), Vec4::new(1.0, 1.0, 1.0, 1.0)),
            (Vec3::new(2.2, 0.0, 0.0), Vec4::new(0.7, 0.7, 0.7, 1.0)),
            (Vec3::new(2.2, 0.0, -0.8), Vec4::new(0.4, 0.4, 0.4, 1.0)),
        ];

        for (pos, tint) in reds {
            let id = zone.insert(WorldTransform {
                position: SimPos::from_vec3(pos),
                facing: Facing::ZERO,
            });
            zone.components_mut().insert(id, MaterialInstanceId::Red);
            zone.components_mut().insert(id, Tint(tint));
        }
        for (pos, tint) in golds {
            let id = zone.insert(WorldTransform {
                position: SimPos::from_vec3(pos),
                facing: Facing::ZERO,
            });
            zone.components_mut().insert(id, MaterialInstanceId::Gold);
            zone.components_mut().insert(id, Tint(tint));
        }

        let _ = zid;
        Self { zones }
    }
}

impl Simulation for Game {
    type Command = ();
    fn tick(&mut self, _: std::time::Duration) {}
}

// --- Mesh data ----------------------------------------------------------

#[rustfmt::skip]
const CUBE_POSITIONS: &[[f32; 3]] = &[
    [-0.35, -0.35, -0.35],
    [ 0.35, -0.35, -0.35],
    [ 0.35,  0.35, -0.35],
    [-0.35,  0.35, -0.35],
    [-0.35, -0.35,  0.35],
    [ 0.35, -0.35,  0.35],
    [ 0.35,  0.35,  0.35],
    [-0.35,  0.35,  0.35],
];
#[rustfmt::skip]
const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // -Z (bottom)
    4, 6, 5, 4, 7, 6, // +Z (top)
    0, 3, 7, 0, 7, 4, // -X
    1, 5, 6, 1, 6, 2, // +X
    3, 2, 6, 3, 6, 7, // +Y
    0, 4, 5, 0, 5, 1, // -Y
];

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 64;

struct Materials {
    camera: Camera,
    camera_binding: CameraBinding,
    material: UnlitColoredMaterial,
    instances: MaterialInstanceRegistry<UnlitColoredInstance, MaterialInstanceId>,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    buckets: InstanceBuckets<MaterialInstanceId, MeshInstanceAttribs>,
    started: Instant,
}

impl View for Materials {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — materials demo",
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        use wgpu::util::DeviceExt;

        let device = &renderer.device;
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(device);
        let material = UnlitColoredMaterial::new(renderer, camera_binding.layout());

        let mut instances = MaterialInstanceRegistry::new();
        instances.register(
            MaterialInstanceId::Red,
            material.create_instance(renderer, Vec4::new(0.85, 0.20, 0.20, 1.0)),
        );
        instances.register(
            MaterialInstanceId::Gold,
            material.create_instance(renderer, Vec4::new(0.90, 0.72, 0.25, 1.0)),
        );

        let cube_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube positions"),
            contents: bytemuck::cast_slice(CUBE_POSITIONS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube indices"),
            contents: bytemuck::cast_slice(CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut buckets = InstanceBuckets::<MaterialInstanceId, MeshInstanceAttribs>::new(
            "material instance attribs",
            MAX_INSTANCES,
        );
        buckets.register(device, MaterialInstanceId::Red);
        buckets.register(device, MaterialInstanceId::Gold);

        Self {
            camera,
            camera_binding,
            material,
            instances,
            cube_vertices,
            cube_indices,
            buckets,
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

        // Wall-clock orbit so motion continues even with no sim tick.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 6.5;
        let angle = t * 0.35;
        self.camera.position = Vec3::new(angle.sin() * radius, angle.cos() * radius, 1.5);
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Extract: bucket per-instance attribs by material id.
        self.buckets.begin_frame();
        for (_, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&mid) = zone.components().get::<MaterialInstanceId>(id) else {
                    continue;
                };
                let tint = zone
                    .components()
                    .get::<Tint>(id)
                    .map(|t| t.0)
                    .unwrap_or(Vec4::ONE);
                let model =
                    Mat4::from_rotation_translation(obj.facing.to_quat(), obj.position.to_vec3());
                self.buckets
                    .push(mid, MeshInstanceAttribs::new(model, tint));
            }
        }
        self.buckets.upload(&renderer.queue);

        // Draw: one instanced call per non-empty bucket.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.cube_vertices.slice(..));
        pass.set_index_buffer(self.cube_indices.slice(..), wgpu::IndexFormat::Uint16);
        for (mid, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(instance) = self.instances.get(*mid) else {
                continue;
            };
            pass.set_bind_group(1, instance.bind_group(), &[]);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..count);
        }
    }

    fn input(
        &mut self,
        _: &Game,
        ctx: &mut EngineCtx,
        _: &mut CommandQueue<()>,
        event: &WindowEvent,
    ) {
        let WindowEvent::KeyboardInput { event, .. } = event else {
            return;
        };
        if event.state != ElementState::Pressed {
            return;
        }
        if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
            ctx.event_loop.exit();
        }
    }
}

fn main() {
    currawong::run::<Materials>(Game::new());
}
