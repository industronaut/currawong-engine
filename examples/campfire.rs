//! Campfire: a sim object whose visual is mesh + particle emitters.
//!
//! Demonstrates step 3 of the visual-system architecture — declarative emitter
//! attachment with view-side reconciliation:
//!
//! - Sim holds only state: `Campfire { lit: bool }`. No visual concepts.
//! - The View's `extract_visuals` returns one or more `VisualPart`s per
//!   object. For a campfire that's a log mesh plus, when lit, a flame
//!   emitter and a smoke emitter (each declared at a `(WorldObjectRef, slot)`
//!   key so a single object can carry several emitters).
//! - The View maintains a persistent registry of live emitters. Each frame:
//!   walk sim and declare → emitters not declared this frame stop spawning
//!   but their existing particles linger until they die naturally → orphans
//!   whose `WorldObjectRef` no longer resolves are also stopped.
//! - Lit state changes propagate automatically: extinguish the fire and the
//!   flame emitter is no longer declared, so it stops spawning new particles
//!   and the existing flames fade out over their lifetime.
//!
//! The particle pool, emitter template, and billboard render pipeline are
//! all local to this example for now. Once the shape is settled, the engine
//! will grow an `EmitterReconciler<E, S>` helper that subsumes the
//! reconcile/integrate loop, mirroring how `InstanceBuckets` subsumed the
//! mesh batching loop.
//!
//! Controls: Space toggles fire. 0 pause / 1/2/3 sim speed. Esc to quit.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, EngineCtx, InstanceBuckets, Renderer, Simulation, View, WorldObject, WorldObjectId,
    WorldObjectRef, Zone, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Simulation ----------------------------------------------------------

/// Sim-side campfire state. Only `lit` is a sim fact — the View decides what
/// "lit" looks like (flames, smoke, light) on its own.
struct Campfire {
    lit: bool,
}

struct Game {
    zones: Zones,
    fire_ref: WorldObjectRef,
    elapsed: Duration,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let main_zone = zones.insert(Zone::new());
        let zone = zones.get_mut(main_zone).expect("just inserted");

        let fire_id = zone.insert(WorldObject {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        });
        zone.components_mut()
            .insert(fire_id, Campfire { lit: true });

        Self {
            zones,
            fire_ref: WorldObjectRef {
                zone: main_zone,
                id: fire_id,
            },
            elapsed: Duration::ZERO,
        }
    }

    fn toggle_fire(&mut self) {
        let Some(zone) = self.zones.get_mut(self.fire_ref.zone) else {
            return;
        };
        if let Some(c) = zone.components_mut().get_mut::<Campfire>(self.fire_ref.id) {
            c.lit = !c.lit;
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        self.elapsed += dt;
    }
}

// --- Visual handles ------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MeshId {
    Log,
}

/// Identifies an emitter *template* (the recipe — particle look + behaviour).
/// One template can drive many live emitters across many sim objects.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum EmitterId {
    Flame,
    Smoke,
}

/// Identifies which emitter "slot" on a parent object an attachment fills,
/// so one sim object can carry several emitters keyed independently.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum EmitterSlot {
    Flame,
    Smoke,
}

/// One visual contribution from a sim object. The extract function returns
/// zero or more of these per object; the View dispatches on the variant.
enum VisualPart {
    Mesh {
        mesh_id: MeshId,
        transform: Mat4,
    },
    AttachEmitter {
        template: EmitterId,
        slot: EmitterSlot,
        transform: Mat4,
    },
}

// --- The bridge: sim state → visual parts --------------------------------

fn extract_visuals(
    zone: &Zone,
    id: WorldObjectId,
    obj: &WorldObject,
    push: &mut dyn FnMut(VisualPart),
) {
    if let Some(fire) = zone.components().get::<Campfire>(id) {
        // The log itself is always present, lit or not.
        push(VisualPart::Mesh {
            mesh_id: MeshId::Log,
            transform: Mat4::from_rotation_translation(obj.rotation, obj.position),
        });
        if fire.lit {
            // Flame at the top of the log.
            let flame_origin = obj.position + Vec3::new(0.0, 0.45, 0.0);
            push(VisualPart::AttachEmitter {
                template: EmitterId::Flame,
                slot: EmitterSlot::Flame,
                transform: Mat4::from_translation(flame_origin),
            });
            // Smoke a bit higher, drifting up and slightly off-axis.
            let smoke_origin = obj.position + Vec3::new(0.0, 0.85, 0.0);
            push(VisualPart::AttachEmitter {
                template: EmitterId::Smoke,
                slot: EmitterSlot::Smoke,
                transform: Mat4::from_translation(smoke_origin),
            });
        }
    }
}

// --- Mesh data -----------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    col: [f32; 3],
}
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

const fn v(p: [f32; 3], c: [f32; 3]) -> Vertex {
    Vertex { pos: p, col: c }
}

// A short cylinder-ish "log" — really just a tall thin rectangular prism for
// this demo, in dark woody tones.
#[rustfmt::skip]
const LOG_VERTS: &[Vertex] = &[
    v([-0.40, 0.00, -0.40], [0.30, 0.18, 0.10]),
    v([ 0.40, 0.00, -0.40], [0.32, 0.20, 0.12]),
    v([ 0.40, 0.40, -0.40], [0.40, 0.26, 0.16]),
    v([-0.40, 0.40, -0.40], [0.38, 0.24, 0.14]),
    v([-0.40, 0.00,  0.40], [0.30, 0.18, 0.10]),
    v([ 0.40, 0.00,  0.40], [0.32, 0.20, 0.12]),
    v([ 0.40, 0.40,  0.40], [0.40, 0.26, 0.16]),
    v([-0.40, 0.40,  0.40], [0.38, 0.24, 0.14]),
];
#[rustfmt::skip]
const LOG_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    0, 3, 7, 0, 7, 4,
    1, 5, 6, 1, 6, 2,
    3, 2, 6, 3, 6, 7,
    0, 4, 5, 0, 5, 1,
];

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl Mesh {
    fn new(device: &wgpu::Device, label: &str, verts: &[Vertex], indices: &[u16]) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} vertices")),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} indices")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

// Quad in [-0.5, 0.5]², used for camera-facing billboard particles. Vertex
// is just the 2D corner offset; the shader expands it using camera right/up.
#[repr(C)]
#[derive(Clone, Copy)]
struct QuadCorner {
    offset: [f32; 2],
}
unsafe impl bytemuck::Pod for QuadCorner {}
unsafe impl bytemuck::Zeroable for QuadCorner {}

#[rustfmt::skip]
const QUAD_CORNERS: &[QuadCorner] = &[
    QuadCorner { offset: [-0.5, -0.5] },
    QuadCorner { offset: [ 0.5, -0.5] },
    QuadCorner { offset: [ 0.5,  0.5] },
    QuadCorner { offset: [-0.5,  0.5] },
];
const QUAD_INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

// --- Particle system (CPU-side state, view-only) -------------------------

#[derive(Clone, Copy)]
struct Particle {
    position: Vec3,
    velocity: Vec3,
    age: f32,      // seconds
    lifetime: f32, // seconds
    base_size: f32,
    base_color: Vec4,
}

/// Static recipe shared by all live emitters that reference it.
struct EmitterTemplate {
    spawn_rate: f32,        // particles per second
    particle_lifetime: f32, // seconds
    initial_speed: f32,     // m/s
    speed_jitter: f32,      // ±range on initial speed
    direction: Vec3,        // unit vector; particles emit roughly along this
    cone_half_angle: f32,   // radians; 0 = exact direction, π/2 = hemisphere
    base_size: f32,
    end_size: f32, // size at end of lifetime
    base_color: Vec4,
    end_color: Vec4, // colour at end of lifetime (alpha typically fades to 0)
}

/// One live, parent-attached emitter. Owned by the View, reconciled each
/// frame against what `extract_visuals` declares.
struct LiveEmitter {
    template_id: EmitterId,
    /// Last transform declared for this emitter. Particles emit at this
    /// transform's translation; once spawned they evolve independently in
    /// world space (so the smoke trail stays put if the campfire moves).
    transform: Mat4,
    /// Set true at the start of each frame, set true again if declared, and
    /// checked during integration. A non-declared emitter stops spawning
    /// (linger semantics) but keeps ticking its existing particles.
    declared_this_frame: bool,
    /// Once true, no new spawns. Set when the emitter wasn't declared OR
    /// the parent's `WorldObjectRef` no longer resolves.
    extinguished: bool,
    spawn_accumulator: f32,
    particles: Vec<Particle>,
}

impl LiveEmitter {
    fn new(template_id: EmitterId, transform: Mat4) -> Self {
        Self {
            template_id,
            transform,
            declared_this_frame: true,
            extinguished: false,
            spawn_accumulator: 0.0,
            particles: Vec::new(),
        }
    }
}

/// Per-particle data sent to the GPU. The shader expands each instance into
/// a camera-facing quad using the camera's right/up vectors.
#[repr(C)]
#[derive(Clone, Copy)]
struct ParticleInstance {
    position: [f32; 3],
    size: f32,
    color: [f32; 4],
}
unsafe impl bytemuck::Pod for ParticleInstance {}
unsafe impl bytemuck::Zeroable for ParticleInstance {}

/// Camera uniform: view-projection plus right/up vectors so the particle
/// vertex shader can build camera-facing quads from corner offsets.
#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniform {
    view_proj: Mat4,
    right: Vec4, // xyz = camera right, w unused (padding)
    up: Vec4,    // xyz = camera up, w unused (padding)
}
unsafe impl bytemuck::Pod for CameraUniform {}
unsafe impl bytemuck::Zeroable for CameraUniform {}

// --- Tiny xorshift RNG (deterministic-enough for visuals) ----------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / u64::MAX as f64) as f32
    }
    /// Uniform in `[-1, 1)`.
    fn bipolar(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

// --- Shaders -------------------------------------------------------------

const MESH_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) col: vec3<f32>,
    @location(2) m0: vec4<f32>,
    @location(3) m1: vec4<f32>,
    @location(4) m2: vec4<f32>,
    @location(5) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(in.pos, 1.0);
    var out: VsOut;
    out.clip = camera.view_proj * world;
    out.col = in.col;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.col, 1.0);
}
"#;

const PARTICLE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) corner: vec2<f32>,
    @location(1) particle_pos: vec3<f32>,
    @location(2) size: f32,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world_pos = in.particle_pos
        + camera.right.xyz * in.corner.x * in.size
        + camera.up.xyz * in.corner.y * in.size;
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    out.uv = in.corner * 2.0;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    let mask = smoothstep(1.0, 0.4, r);
    return vec4<f32>(in.color.rgb, in.color.a * mask);
}
"#;

const MAX_INSTANCES_PER_MESH: u32 = 64;
const MAX_PARTICLES_PER_TEMPLATE: u32 = 1024;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// --- View ----------------------------------------------------------------

struct CampfireView {
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    mesh_pipeline: wgpu::RenderPipeline,
    meshes: HashMap<MeshId, Mesh>,
    mesh_instances: InstanceBuckets<MeshId, Mat4>,

    particle_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    particle_instances: InstanceBuckets<EmitterId, ParticleInstance>,

    emitter_templates: HashMap<EmitterId, EmitterTemplate>,
    live_emitters: HashMap<(WorldObjectRef, EmitterSlot), LiveEmitter>,

    rng: Rng,
    started: Instant,
    last_frame: Instant,
}

impl CampfireView {
    fn build_emitter_templates() -> HashMap<EmitterId, EmitterTemplate> {
        let mut t = HashMap::new();
        t.insert(
            EmitterId::Flame,
            EmitterTemplate {
                spawn_rate: 60.0,
                particle_lifetime: 0.7,
                initial_speed: 1.4,
                speed_jitter: 0.5,
                direction: Vec3::Y,
                cone_half_angle: 0.45,
                base_size: 0.18,
                end_size: 0.05,
                base_color: Vec4::new(1.0, 0.65, 0.15, 0.95),
                end_color: Vec4::new(0.6, 0.05, 0.0, 0.0),
            },
        );
        t.insert(
            EmitterId::Smoke,
            EmitterTemplate {
                spawn_rate: 14.0,
                particle_lifetime: 2.5,
                initial_speed: 0.45,
                speed_jitter: 0.15,
                direction: Vec3::new(0.05, 1.0, 0.0).normalize(),
                cone_half_angle: 0.55,
                base_size: 0.20,
                end_size: 0.85,
                base_color: Vec4::new(0.45, 0.45, 0.45, 0.55),
                end_color: Vec4::new(0.25, 0.25, 0.25, 0.0),
            },
        );
        t
    }

    fn upload_camera(&self, queue: &wgpu::Queue) {
        let view_proj = self.camera.view_proj();
        // Camera basis vectors derived from view matrix. The inverse of a
        // pure rotation+translation view matrix has the camera's world-space
        // axes in its first three columns.
        let view_inv = self.camera.view_matrix().inverse();
        let right = view_inv.x_axis.truncate();
        let up = view_inv.y_axis.truncate();
        let uniform = CameraUniform {
            view_proj,
            right: Vec4::new(right.x, right.y, right.z, 0.0),
            up: Vec4::new(up.x, up.y, up.z, 0.0),
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Reconcile pass: walk sim, dispatch each VisualPart into the right
    /// destination. Mesh parts go straight into the instance buckets;
    /// emitter parts update the live emitter registry (lazy create, mark
    /// declared, refresh transform).
    fn extract_and_reconcile(&mut self, sim: &Game) {
        // Mark all live emitters as not-declared at the start of the frame.
        for emitter in self.live_emitters.values_mut() {
            emitter.declared_this_frame = false;
        }
        self.mesh_instances.begin_frame();

        for (zone_id, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let world_ref = WorldObjectRef { zone: zone_id, id };
                extract_visuals(zone, id, obj, &mut |part| match part {
                    VisualPart::Mesh { mesh_id, transform } => {
                        self.mesh_instances.push(mesh_id, transform);
                    }
                    VisualPart::AttachEmitter {
                        template,
                        slot,
                        transform,
                    } => {
                        let key = (world_ref, slot);
                        let emitter = self
                            .live_emitters
                            .entry(key)
                            .or_insert_with(|| LiveEmitter::new(template, transform));
                        emitter.template_id = template;
                        emitter.transform = transform;
                        emitter.declared_this_frame = true;
                        emitter.extinguished = false;
                    }
                });
            }
        }
    }

    /// Integrate live emitters: spawn new particles (if not extinguished),
    /// advance existing ones, drop dead. Drop emitters that have no
    /// declaration AND no live particles left.
    fn integrate_emitters(&mut self, zones: &Zones, dt: f32) {
        let templates = &self.emitter_templates;
        let rng = &mut self.rng;
        self.live_emitters.retain(|key, emitter| {
            // If parent no longer resolves, the emitter is also extinguished.
            let parent_alive = key.0.resolve(zones).is_some();
            if !emitter.declared_this_frame || !parent_alive {
                emitter.extinguished = true;
            }

            let template = templates
                .get(&emitter.template_id)
                .expect("emitter references a registered template");

            // Tick existing particles.
            for p in emitter.particles.iter_mut() {
                p.age += dt;
                p.position += p.velocity * dt;
            }
            emitter.particles.retain(|p| p.age < p.lifetime);

            // Spawn new particles unless extinguished. Spawn count derived
            // from rate × dt with fractional carry-over so low rates still
            // emit at the right average frequency.
            if !emitter.extinguished {
                emitter.spawn_accumulator += template.spawn_rate * dt;
                while emitter.spawn_accumulator >= 1.0
                    && (emitter.particles.len() as u32) < MAX_PARTICLES_PER_TEMPLATE
                {
                    emitter.spawn_accumulator -= 1.0;
                    emitter
                        .particles
                        .push(spawn_particle(template, emitter.transform, rng));
                }
            } else {
                // Don't accumulate while extinguished.
                emitter.spawn_accumulator = 0.0;
            }

            // Keep emitter if it could still spawn or has live particles.
            !emitter.extinguished || !emitter.particles.is_empty()
        });
    }

    /// Build the per-emitter-template particle instance buckets from live
    /// state. This is the equivalent of "extract" for particles, except the
    /// data comes from view-side state (the particle pool) not from sim.
    fn build_particle_buckets(&mut self) {
        self.particle_instances.begin_frame();
        for emitter in self.live_emitters.values() {
            let template = self
                .emitter_templates
                .get(&emitter.template_id)
                .expect("template registered");
            for p in &emitter.particles {
                let life = (p.age / p.lifetime).clamp(0.0, 1.0);
                let size = lerp(p.base_size, template.end_size, life);
                let color = p.base_color.lerp(template.end_color, life);
                self.particle_instances.push(
                    emitter.template_id,
                    ParticleInstance {
                        position: [p.position.x, p.position.y, p.position.z],
                        size,
                        color: [color.x, color.y, color.z, color.w],
                    },
                );
            }
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn spawn_particle(template: &EmitterTemplate, transform: Mat4, rng: &mut Rng) -> Particle {
    // Random direction in a cone around `template.direction`.
    let dir = jitter_direction(template.direction, template.cone_half_angle, rng);
    let speed = template.initial_speed + rng.bipolar() * template.speed_jitter;
    let position = transform.transform_point3(Vec3::ZERO);
    Particle {
        position,
        velocity: dir * speed.max(0.0),
        age: 0.0,
        lifetime: template.particle_lifetime,
        base_size: template.base_size,
        base_color: template.base_color,
    }
}

/// Random unit vector within `half_angle` radians of `axis`.
fn jitter_direction(axis: Vec3, half_angle: f32, rng: &mut Rng) -> Vec3 {
    if half_angle <= 0.0 {
        return axis.normalize();
    }
    // Pick a tangent basis to `axis`.
    let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let tangent = axis.cross(helper).normalize();
    let bitangent = axis.cross(tangent);
    // Cone sample: angle from axis uniform in [0, half_angle), azimuth in [0, 2π).
    let theta = rng.unit() * half_angle;
    let phi = rng.unit() * std::f32::consts::TAU;
    let st = theta.sin();
    let ct = theta.cos();
    (axis * ct + tangent * (st * phi.cos()) + bitangent * (st * phi.sin())).normalize()
}

impl View for CampfireView {
    type Sim = Game;

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("campfire pipeline layout"),
            bind_group_layouts: &[Some(&camera_bgl)],
            ..Default::default()
        });

        let mesh_pipeline = build_mesh_pipeline(device, &layout, renderer.surface_format());
        let particle_pipeline = build_particle_pipeline(device, &layout, renderer.surface_format());

        let mut meshes = HashMap::new();
        meshes.insert(
            MeshId::Log,
            Mesh::new(device, "log", LOG_VERTS, LOG_INDICES),
        );

        let mut mesh_instances =
            InstanceBuckets::<MeshId, Mat4>::new("mesh instances", MAX_INSTANCES_PER_MESH);
        mesh_instances.register(device, MeshId::Log);

        // Quad geometry shared by all particle billboards.
        use wgpu::util::DeviceExt;
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad vertices"),
            contents: bytemuck::cast_slice(QUAD_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad indices"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut particle_instances = InstanceBuckets::<EmitterId, ParticleInstance>::new(
            "particle instances",
            MAX_PARTICLES_PER_TEMPLATE,
        );
        for &k in &[EmitterId::Flame, EmitterId::Smoke] {
            particle_instances.register(device, k);
        }

        Self {
            camera: Camera {
                position: Vec3::new(0.0, 1.6, 4.0),
                target: Vec3::new(0.0, 0.6, 0.0),
                ..Camera::default()
            },
            camera_buffer,
            camera_bind_group,

            mesh_pipeline,
            meshes,
            mesh_instances,

            particle_pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            particle_instances,

            emitter_templates: Self::build_emitter_templates(),
            live_emitters: HashMap::new(),

            rng: Rng::new(0xC0FFEE),
            started: Instant::now(),
            last_frame: Instant::now(),
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

        // Slow camera orbit around the campfire on wall clock.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 4.5;
        let angle = t * 0.25;
        self.camera.position = Vec3::new(angle.sin() * radius, 1.8, angle.cos() * radius);
        self.camera.target = Vec3::new(0.0, 0.6, 0.0);

        self.upload_camera(&renderer.queue);

        // Wall-clock dt for emitter integration. (We could route sim_time
        // through the Game and use that instead — left as a future toggle.)
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // 1. Walk sim, declare visuals: mesh parts → mesh_instances bucket;
        //    emitter parts → live_emitter reconciliation.
        self.extract_and_reconcile(sim);
        // 2. Tick particles + spawn new ones (or stop spawning if undeclared).
        self.integrate_emitters(&sim.zones, dt);
        // 3. Build per-template particle instance buffers from live state.
        self.build_particle_buckets();

        // 4. Upload all instance buffers in one pass.
        self.mesh_instances.upload(&renderer.queue);
        self.particle_instances.upload(&renderer.queue);

        // 5. Draw opaque meshes.
        pass.set_pipeline(&self.mesh_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (mesh_id, instance_buffer, count) in self.mesh_instances.iter_filled() {
            let mesh = &self.meshes[&mesh_id];
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }

        // 6. Draw particle billboards. Same camera bind group; different
        //    pipeline (alpha blending, no depth write).
        pass.set_pipeline(&self.particle_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (_emitter_id, instance_buffer, count) in self.particle_instances.iter_filled() {
            pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..count);
        }
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
            KeyCode::Space => sim.toggle_fire(),
            KeyCode::Digit0 => ctx.clock.set_speed(0.0),
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            KeyCode::Digit2 => ctx.clock.set_speed(2.0),
            KeyCode::Digit3 => ctx.clock.set_speed(3.0),
            _ => {}
        }
    }

    fn title() -> &'static str {
        "currawong — campfire (space toggles fire)"
    }

    fn depth_format() -> Option<wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }
}

// --- Pipeline construction (split out to keep View::init readable) -------

fn build_mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Mat4>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 32,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 48,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn build_particle_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadCorner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: 16,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // Particles read depth (so they're occluded by the log) but
            // don't write depth (so transparent overlap blends correctly).
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn main() {
    currawong::run::<CampfireView>(Game::new());
}
