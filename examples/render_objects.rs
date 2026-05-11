//! Sim → `RenderId` → `RenderTemplate` → mesh parts → instanced draws.
//!
//! Two render templates are registered at init:
//! - `Campfire` — two parts (log + stone base), two materials.
//! - `Stake` — one part, single material.
//!
//! Sim has five objects, each tagged with a `RenderId` component. Every frame:
//! 1. Walk the sim, look up each object's template by `RenderId`.
//! 2. For each part in the template, compose `world = object_xform *
//!    part.local_transform` and push an [`UnlitColoredAttribs`] into the
//!    bucket keyed by `(mesh, material)`.
//! 3. One indexed-instanced draw per non-empty bucket.
//!
//! Shows: many sim objects share one template; one template fans out into
//! multiple draw parts; parts of different templates can share material
//! instances and accumulate into the same bucket.

use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, InstanceBuckets, MaterialInstanceRegistry, RenderRegistry,
    RenderTemplate, Renderer, Simulation, UnlitColoredAttribs, UnlitColoredInstance,
    UnlitColoredMaterial, View, WorldObject, Zone, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim-side identifiers -----------------------------------------------

/// Names a render template. Carried as a component on sim objects;
/// the View resolves it to a [`RenderTemplate`] via its registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RenderId {
    Campfire,
    Stake,
}

// --- View-side identifiers ----------------------------------------------

/// User-defined mesh handle. The engine doesn't own meshes; the View keeps
/// a `HashMap<MeshHandle, ...>` of GPU buffers and looks them up at draw.
/// Only one mesh in this demo — different parts scale it via local
/// transforms.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MeshHandle {
    Cube,
}

/// User-defined material-instance key. Two material instances of the same
/// [`UnlitColoredMaterial`] (template), differing in base colour.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MatKey {
    Wood,
    Stone,
}

/// Bucket key: meshes are bound per (mesh, material) pair so the draw loop
/// can issue one instanced call per pair.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct BucketKey {
    mesh: MeshHandle,
    material: MatKey,
}

// --- Sim ----------------------------------------------------------------

struct Game {
    zones: Zones,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zid = zones.insert(Zone::new());
        let zone = zones.get_mut(zid).expect("just inserted");

        // Three campfires along X.
        for x in [-3.0, 0.0, 3.0] {
            let id = zone.insert(WorldObject {
                position: Vec3::new(x, 0.0, 0.0),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, RenderId::Campfire);
        }
        // Two stakes along Z, offset back so they're behind the campfires.
        for z in [-2.5, 2.5] {
            let id = zone.insert(WorldObject {
                position: Vec3::new(0.0, 0.0, z),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, RenderId::Stake);
        }
        Self { zones }
    }
}

impl Simulation for Game {
    fn tick(&mut self, _: std::time::Duration) {}
}

// --- Mesh data (one unit cube; scaled per part) -------------------------

#[rustfmt::skip]
const CUBE_POSITIONS: &[[f32; 3]] = &[
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
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

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 64;

struct Demo {
    camera: Camera,
    camera_binding: CameraBinding,
    material: UnlitColoredMaterial,
    instances: MaterialInstanceRegistry<UnlitColoredInstance, MatKey>,
    templates: RenderRegistry<RenderId, MeshHandle, MatKey>,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    buckets: InstanceBuckets<BucketKey, UnlitColoredAttribs>,
    started: Instant,
}

impl View for Demo {
    type Sim = Game;

    fn init(renderer: &Renderer) -> Self {
        use wgpu::util::DeviceExt;

        let device = &renderer.device;
        let camera = Camera::default();
        let camera_binding = CameraBinding::new(device);
        let material = UnlitColoredMaterial::new(renderer, camera_binding.layout());

        let mut instances = MaterialInstanceRegistry::new();
        instances.register(
            MatKey::Wood,
            material.create_instance(renderer, Vec4::new(0.55, 0.32, 0.18, 1.0)),
        );
        instances.register(
            MatKey::Stone,
            material.create_instance(renderer, Vec4::new(0.55, 0.55, 0.58, 1.0)),
        );

        // Two render templates. The Campfire fans out into a log (long, low)
        // and a stone base (wide, flatter, sits underneath). The Stake is a
        // single tall column.
        let mut templates: RenderRegistry<RenderId, MeshHandle, MatKey> = RenderRegistry::new();
        templates.register(
            RenderId::Campfire,
            RenderTemplate::new("campfire")
                // Log: stretched along X, sitting just above the ground.
                .with_mesh_part(
                    MeshHandle::Cube,
                    MatKey::Wood,
                    Mat4::from_scale_rotation_translation(
                        Vec3::new(1.4, 0.35, 0.35),
                        Quat::IDENTITY,
                        Vec3::new(0.0, 0.4, 0.0),
                    ),
                )
                // Base: flat slab beneath the log.
                .with_mesh_part(
                    MeshHandle::Cube,
                    MatKey::Stone,
                    Mat4::from_scale_rotation_translation(
                        Vec3::new(1.8, 0.18, 1.0),
                        Quat::IDENTITY,
                        Vec3::new(0.0, 0.0, 0.0),
                    ),
                ),
        );
        templates.register(
            RenderId::Stake,
            RenderTemplate::new("stake").with_mesh_part(
                MeshHandle::Cube,
                MatKey::Wood,
                Mat4::from_scale_rotation_translation(
                    Vec3::new(0.25, 1.6, 0.25),
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.8, 0.0),
                ),
            ),
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

        // Buckets per (mesh, material) pair that appears in any template.
        let mut buckets = InstanceBuckets::<BucketKey, UnlitColoredAttribs>::new(
            "render-object attribs",
            MAX_INSTANCES,
        );
        for material in [MatKey::Wood, MatKey::Stone] {
            buckets.register(
                device,
                BucketKey {
                    mesh: MeshHandle::Cube,
                    material,
                },
            );
        }

        Self {
            camera,
            camera_binding,
            material,
            instances,
            templates,
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

        // Wall-clock orbit.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 7.5;
        let angle = t * 0.35;
        self.camera.position = Vec3::new(angle.sin() * radius, 2.5, angle.cos() * radius);
        self.camera.target = Vec3::ZERO;
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Extract: walk sim, look up template, fan out into parts, bucket.
        self.buckets.begin_frame();
        for (_, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&rid) = zone.components().get::<RenderId>(id) else {
                    continue;
                };
                let Some(template) = self.templates.get(rid) else {
                    continue;
                };
                let object_xform = Mat4::from_rotation_translation(obj.rotation, obj.position);
                for part in template.mesh_parts() {
                    let world = object_xform * part.local_transform;
                    self.buckets.push(
                        BucketKey {
                            mesh: part.mesh,
                            material: part.material,
                        },
                        UnlitColoredAttribs::new(world, Vec4::ONE),
                    );
                }
            }
        }
        self.buckets.upload(&renderer.queue);

        // Draw: one indexed-instanced call per non-empty bucket.
        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.cube_vertices.slice(..));
        pass.set_index_buffer(self.cube_indices.slice(..), wgpu::IndexFormat::Uint16);
        for (bucket_key, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(instance) = self.instances.get(bucket_key.material) else {
                continue;
            };
            pass.set_bind_group(1, instance.bind_group(), &[]);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..count);
        }
    }

    fn input(&mut self, _: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
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

    fn title() -> &'static str {
        "currawong — render objects demo"
    }

    fn depth_format() -> Option<wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }
}

fn main() {
    currawong::run::<Demo>(Game::new());
}
