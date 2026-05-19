//! Campfire: a sim object whose visual is mesh + particle emitters.
//!
//! Demonstrates declarative emitter attachment with engine-managed
//! reconciliation via [`EmitterReconciler`]:
//!
//! - Sim holds only state: `Campfire { lit: bool }`. No visual concepts.
//! - The View's `extract_visuals` returns one or more `VisualPart`s per
//!   object. For a campfire that's a log mesh plus, when lit, a flame
//!   emitter and a smoke emitter (each declared at a `(WorldObjectRef, slot)`
//!   key so one sim object can carry several emitters).
//! - `EmitterReconciler` owns the live emitter registry, particle pool,
//!   and lifecycle. Each frame the View calls `begin_frame` → `declare` per
//!   emitter attachment → `integrate(dt)` → `iter_particles()` for the
//!   render extract.
//! - Lit-state changes propagate automatically: extinguish the fire and the
//!   flame emitter is no longer declared, so it stops spawning new
//!   particles and the existing flames linger out per their template's
//!   `ParticleLifecycle::Linger` setting.
//!
//! Particle visual interpolation (size and colour curves) lives in this
//! example as a `HashMap<EmitterId, ParticleVisual>` — the engine helper
//! deliberately doesn't model visuals, only kinematics and lifecycle, so
//! the user's render code is free to choose how particles look.
//!
//! Controls: Space toggles fire. 0 pause / 1/2/3 sim speed. Esc to quit.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, CommandQueue, EmitterReconciler, EmitterTemplate, EngineCtx, Facing,
    InstanceBuckets, ParticleLifecycle, Renderer, SimPos, Simulation, View, ViewConfig,
    WorldObjectId, WorldObjectRef, WorldTransform, Zone, Zones, mat4_instance_attributes, wgpu,
    winit,
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

        let fire_id = zone.insert(WorldTransform {
            position: SimPos::ZERO,
            facing: Facing::ZERO,
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

/// External sim mutations the View can request. Today just one: the user
/// pressing Space asks for the fire to toggle. `apply_command` resolves
/// `fire_ref` against current sim state, so the command is primitive (no
/// reference, no closure).
#[derive(Debug, Clone)]
enum Command {
    ToggleFire,
}

impl Simulation for Game {
    type Command = Command;

    fn tick(&mut self, dt: Duration) {
        self.elapsed += dt;
    }

    fn apply_command(&mut self, cmd: &Command) {
        match cmd {
            Command::ToggleFire => self.toggle_fire(),
        }
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
    obj: &WorldTransform,
    push: &mut dyn FnMut(VisualPart),
) {
    if let Some(fire) = zone.components().get::<Campfire>(id) {
        // Convert the sim transform to a `glam::Mat4` at the extract seam.
        let world_pos = obj.position.to_vec3();
        // The log itself is always present, lit or not.
        push(VisualPart::Mesh {
            mesh_id: MeshId::Log,
            transform: Mat4::from_rotation_translation(obj.facing.to_quat(), world_pos),
        });
        if fire.lit {
            // Flame at the top of the log.
            let flame_origin = world_pos + Vec3::new(0.0, 0.0, 0.45);
            push(VisualPart::AttachEmitter {
                template: EmitterId::Flame,
                slot: EmitterSlot::Flame,
                transform: Mat4::from_translation(flame_origin),
            });
            // Smoke a bit higher, drifting up and slightly off-axis.
            let smoke_origin = world_pos + Vec3::new(0.0, 0.0, 0.85);
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
// this demo, in dark woody tones. Z-up: rests on z=0, top at z=0.40.
#[rustfmt::skip]
const LOG_VERTS: &[Vertex] = &[
    v([-0.40, -0.40, 0.00], [0.30, 0.18, 0.10]),
    v([ 0.40, -0.40, 0.00], [0.32, 0.20, 0.12]),
    v([ 0.40, -0.40, 0.40], [0.40, 0.26, 0.16]),
    v([-0.40, -0.40, 0.40], [0.38, 0.24, 0.14]),
    v([-0.40,  0.40, 0.00], [0.30, 0.18, 0.10]),
    v([ 0.40,  0.40, 0.00], [0.32, 0.20, 0.12]),
    v([ 0.40,  0.40, 0.40], [0.40, 0.26, 0.16]),
    v([-0.40,  0.40, 0.40], [0.38, 0.24, 0.14]),
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

// --- Particle visual params (size & colour curves, view-only) ------------

/// Per-template visual interpolation that the engine helper doesn't model.
/// The reconciler tracks particle kinematics; this provides the curves the
/// user wants to apply when turning a `Particle` into GPU instance data.
struct ParticleVisual {
    base_size: f32,
    end_size: f32,
    base_color: Vec4,
    end_color: Vec4,
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
    camera_binding: CameraBinding,

    mesh_pipeline: wgpu::RenderPipeline,
    meshes: HashMap<MeshId, Mesh>,
    mesh_instances: InstanceBuckets<MeshId, Mat4>,

    particle_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    particle_instances: InstanceBuckets<EmitterId, ParticleInstance>,

    /// Engine-managed emitter lifecycle + particle integration.
    emitters: EmitterReconciler<EmitterId, EmitterSlot>,
    /// Per-template visual interpolation (size/colour curves) — keyed by
    /// the same `EmitterId` the reconciler uses.
    visuals: HashMap<EmitterId, ParticleVisual>,

    started: Instant,
    last_frame: Instant,
}

impl CampfireView {
    /// Kinematic recipes for the engine reconciler. Visuals (size, colour)
    /// live in [`build_visuals`](Self::build_visuals).
    fn flame_template() -> EmitterTemplate {
        EmitterTemplate {
            spawn_rate: 60.0,
            particle_lifetime: 0.7,
            initial_speed: 1.4,
            speed_jitter: 0.5,
            direction: Vec3::Z,
            cone_half_angle: 0.45,
            max_particles: MAX_PARTICLES_PER_TEMPLATE,
            on_parent_lost: ParticleLifecycle::Linger,
        }
    }

    fn smoke_template() -> EmitterTemplate {
        EmitterTemplate {
            spawn_rate: 14.0,
            particle_lifetime: 2.5,
            initial_speed: 0.45,
            speed_jitter: 0.15,
            direction: Vec3::new(0.05, 0.0, 1.0).normalize(),
            cone_half_angle: 0.55,
            max_particles: MAX_PARTICLES_PER_TEMPLATE,
            on_parent_lost: ParticleLifecycle::Linger,
        }
    }

    fn build_visuals() -> HashMap<EmitterId, ParticleVisual> {
        let mut v = HashMap::new();
        v.insert(
            EmitterId::Flame,
            ParticleVisual {
                base_size: 0.18,
                end_size: 0.05,
                base_color: Vec4::new(1.0, 0.65, 0.15, 0.95),
                end_color: Vec4::new(0.6, 0.05, 0.0, 0.0),
            },
        );
        v.insert(
            EmitterId::Smoke,
            ParticleVisual {
                base_size: 0.20,
                end_size: 0.85,
                base_color: Vec4::new(0.45, 0.45, 0.45, 0.55),
                end_color: Vec4::new(0.25, 0.25, 0.25, 0.0),
            },
        );
        v
    }

    /// Walk sim once: dispatch each `VisualPart` into the right destination.
    /// Mesh parts go straight into the instance buckets; emitter parts are
    /// declared on the engine reconciler.
    fn extract_and_declare(&mut self, sim: &Game) {
        self.mesh_instances.begin_frame();
        self.emitters.begin_frame();
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
                        self.emitters.declare(world_ref, slot, template, transform);
                    }
                });
            }
        }
    }

    /// Walk every live particle, look up its template's visual params, and
    /// push GPU instance data. The reconciler tracks kinematics; this turns
    /// each particle into the size + colour the user wants drawn.
    fn build_particle_buckets(&mut self) {
        self.particle_instances.begin_frame();
        for (template_id, particle) in self.emitters.iter_particles() {
            let v = &self.visuals[&template_id];
            let life = particle.life();
            let size = v.base_size + (v.end_size - v.base_size) * life;
            let color = v.base_color.lerp(v.end_color, life);
            self.particle_instances.push(
                template_id,
                ParticleInstance {
                    position: [
                        particle.position.x,
                        particle.position.y,
                        particle.position.z,
                    ],
                    size,
                    color: [color.x, color.y, color.z, color.w],
                },
            );
        }
    }
}

impl View for CampfireView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — campfire (space toggles fire)",
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let camera_binding = CameraBinding::new(device);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("campfire pipeline layout"),
            bind_group_layouts: &[Some(camera_binding.layout())],
            ..Default::default()
        });

        let mesh_pipeline = build_mesh_pipeline(
            device,
            &layout,
            renderer.surface_format(),
            renderer.id_format(),
        );
        let particle_pipeline = build_particle_pipeline(
            device,
            &layout,
            renderer.surface_format(),
            renderer.id_format(),
        );

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

        let mut emitters = EmitterReconciler::<EmitterId, EmitterSlot>::new(0xC0FFEE);
        emitters.register_template(EmitterId::Flame, Self::flame_template());
        emitters.register_template(EmitterId::Smoke, Self::smoke_template());

        Self {
            camera: Camera {
                position: Vec3::new(0.0, -4.0, 1.6),
                target: Vec3::new(0.0, 0.0, 0.6),
                ..Camera::default()
            },
            camera_binding,

            mesh_pipeline,
            meshes,
            mesh_instances,

            particle_pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            particle_instances,

            emitters,
            visuals: Self::build_visuals(),

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
        self.camera.position = Vec3::new(angle.sin() * radius, angle.cos() * radius, 1.8);
        self.camera.target = Vec3::new(0.0, 0.0, 0.6);

        self.camera_binding.write(&renderer.queue, &self.camera);

        // Wall-clock dt for emitter integration. (We could route sim_time
        // through the Game and use that instead — left as a future toggle.)
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // 1. Walk sim, declare visuals: mesh parts → mesh_instances bucket;
        //    emitter parts → engine reconciler (lazy create / refresh
        //    transform / mark declared).
        self.extract_and_declare(sim);
        // 2. Engine ticks particles + spawns new ones (or stops spawning if
        //    undeclared / parent gone, per template lifecycle).
        self.emitters.integrate(&sim.zones, dt);
        // 3. Build per-template particle instance buffers, applying the
        //    user-side visual interpolation curves to each live particle.
        self.build_particle_buckets();

        // 4. Upload all instance buffers in one pass.
        self.mesh_instances.upload(&renderer.queue);
        self.particle_instances.upload(&renderer.queue);

        // 5. Draw opaque meshes.
        pass.set_pipeline(&self.mesh_pipeline);
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        for (mesh_id, instance_buffer, count) in self.mesh_instances.iter_filled() {
            let mesh = &self.meshes[mesh_id];
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }

        // 6. Draw particle billboards. Same camera bind group; different
        //    pipeline (alpha blending, no depth write).
        pass.set_pipeline(&self.particle_pipeline);
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        for (_emitter_id, instance_buffer, count) in self.particle_instances.iter_filled() {
            pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..count);
        }
    }

    fn input(
        &mut self,
        _sim: &Game,
        ctx: &mut EngineCtx,
        cmds: &mut CommandQueue<Command>,
        event: &WindowEvent,
    ) {
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
            KeyCode::Space => cmds.push_now(Command::ToggleFire),
            KeyCode::Digit0 => ctx.clock.set_speed(0.0),
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            KeyCode::Digit2 => ctx.clock.set_speed(2.0),
            KeyCode::Digit3 => ctx.clock.set_speed(3.0),
            _ => {}
        }
    }
}

// --- Pipeline construction (split out to keep View::init readable) -------

fn build_mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    surface_format: wgpu::TextureFormat,
    id_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
    });
    let mat4_attrs = mat4_instance_attributes(2);
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
                    attributes: &mat4_attrs,
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Opt out of the hit-ID attachment (#56 PR 1): match the
                // format wgpu sees on the attachment but disable writes.
                Some(wgpu::ColorTargetState {
                    format: id_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }),
            ],
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
    id_format: wgpu::TextureFormat,
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
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                // Opt out of the hit-ID attachment (#56 PR 1): match the
                // format wgpu sees on the attachment but disable writes.
                Some(wgpu::ColorTargetState {
                    format: id_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }),
            ],
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
