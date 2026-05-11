//! Sim → `RenderId` → `RenderTemplate` → (mesh parts + emitter parts) → draws.
//!
//! Two render templates are registered at init:
//! - `Campfire` — mesh parts (log + stone base) **and** emitter parts
//!   (flame at top of log, smoke higher up).
//! - `Stake` — single mesh part.
//!
//! Sim has five objects, each tagged with a `RenderId` component. Every frame:
//! 1. Walk the sim. For each object look up its template by `RenderId`.
//! 2. For each `MeshPart`, compose `world = object_xform *
//!    part.local_transform` and push an [`UnlitColoredAttribs`] into the
//!    bucket keyed by `(mesh, material)`.
//! 3. For each `EmitterPart`, declare it on the [`EmitterReconciler`] at
//!    `world = object_xform * part.local_transform`.
//! 4. Integrate the reconciler, build a per-emitter-template particle
//!    bucket, draw meshes then particles.
//!
//! Shows: a single template fans out into multiple draw kinds; emitter
//! parts on a template route into [`EmitterReconciler`] without the sim
//! ever touching an emitter concept.

use std::collections::HashMap;
use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EmitterReconciler, EmitterTemplate, EngineCtx, InstanceBuckets,
    MaterialInstanceRegistry, ParticleLifecycle, RenderRegistry, RenderTemplate, Renderer,
    Simulation, UnlitColoredAttribs, UnlitColoredInstance, UnlitColoredMaterial, View, WorldObject,
    WorldObjectRef, Zone, Zones, wgpu, winit,
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

/// Emitter template id. One template per *kind of effect* (flame, smoke);
/// many live emitters share one template.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum EmitterId {
    Flame,
    Smoke,
}

/// Which emitter "slot" on a parent object an attachment fills. Lets one
/// render template carry several emitters keyed independently — the
/// campfire has both a flame slot and a smoke slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum EmitterSlot {
    Flame,
    Smoke,
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

// --- Particle plumbing (billboard quad expanded by camera basis) --------

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

/// Size/colour interpolation curve per emitter template. The engine
/// reconciler doesn't model visuals; this lives view-side, keyed by the
/// same id the reconciler uses.
struct ParticleVisual {
    base_size: f32,
    end_size: f32,
    base_color: Vec4,
    end_color: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ParticleInstance {
    position: [f32; 3],
    size: f32,
    color: [f32; 4],
}
unsafe impl bytemuck::Pod for ParticleInstance {}
unsafe impl bytemuck::Zeroable for ParticleInstance {}

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

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 64;
const MAX_PARTICLES_PER_TEMPLATE: u32 = 1024;

/// Five generics on `RenderRegistry` adds up — pin them once.
type Templates = RenderRegistry<RenderId, MeshHandle, MatKey, EmitterId, EmitterSlot>;

struct Demo {
    camera: Camera,
    camera_binding: CameraBinding,
    material: UnlitColoredMaterial,
    instances: MaterialInstanceRegistry<UnlitColoredInstance, MatKey>,
    templates: Templates,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    buckets: InstanceBuckets<BucketKey, UnlitColoredAttribs>,

    // Particle path.
    particle_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    particle_instances: InstanceBuckets<EmitterId, ParticleInstance>,
    emitters: EmitterReconciler<EmitterId, EmitterSlot>,
    visuals: HashMap<EmitterId, ParticleVisual>,

    started: Instant,
    last_frame: Instant,
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
        // and a stone base (wide, flatter, sits underneath), plus a flame
        // emitter at the log top and a smoke emitter above. The Stake is a
        // single tall column with no emitters.
        let mut templates: Templates = RenderRegistry::new();
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
                )
                // Flame at the top of the log.
                .with_emitter_part(
                    EmitterId::Flame,
                    EmitterSlot::Flame,
                    Mat4::from_translation(Vec3::new(0.0, 0.65, 0.0)),
                )
                // Smoke a little higher, drifts up.
                .with_emitter_part(
                    EmitterId::Smoke,
                    EmitterSlot::Smoke,
                    Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
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

        // Particle pipeline: reads camera, expands a unit quad per particle
        // using camera right/up basis (alpha-blended, depth-test but no
        // depth-write so overlapping particles blend).
        let particle_pipeline =
            build_particle_pipeline(device, camera_binding.layout(), renderer.surface_format());
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad corners"),
            contents: bytemuck::cast_slice(QUAD_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad indices"),
            contents: bytemuck::cast_slice(QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut particle_instances = InstanceBuckets::<EmitterId, ParticleInstance>::new(
            "particles",
            MAX_PARTICLES_PER_TEMPLATE,
        );
        for emitter in [EmitterId::Flame, EmitterId::Smoke] {
            particle_instances.register(device, emitter);
        }

        let mut emitters = EmitterReconciler::<EmitterId, EmitterSlot>::new(0xCAFEF00D);
        emitters.register_template(
            EmitterId::Flame,
            EmitterTemplate {
                spawn_rate: 60.0,
                particle_lifetime: 0.7,
                initial_speed: 1.4,
                speed_jitter: 0.5,
                direction: Vec3::Y,
                cone_half_angle: 0.45,
                max_particles: MAX_PARTICLES_PER_TEMPLATE,
                on_parent_lost: ParticleLifecycle::Linger,
            },
        );
        emitters.register_template(
            EmitterId::Smoke,
            EmitterTemplate {
                spawn_rate: 14.0,
                particle_lifetime: 2.5,
                initial_speed: 0.45,
                speed_jitter: 0.15,
                direction: Vec3::new(0.05, 1.0, 0.0).normalize(),
                cone_half_angle: 0.55,
                max_particles: MAX_PARTICLES_PER_TEMPLATE,
                on_parent_lost: ParticleLifecycle::Linger,
            },
        );

        let mut visuals = HashMap::new();
        visuals.insert(
            EmitterId::Flame,
            ParticleVisual {
                base_size: 0.18,
                end_size: 0.05,
                base_color: Vec4::new(1.0, 0.65, 0.15, 0.95),
                end_color: Vec4::new(0.6, 0.05, 0.0, 0.0),
            },
        );
        visuals.insert(
            EmitterId::Smoke,
            ParticleVisual {
                base_size: 0.20,
                end_size: 0.85,
                base_color: Vec4::new(0.45, 0.45, 0.45, 0.55),
                end_color: Vec4::new(0.25, 0.25, 0.25, 0.0),
            },
        );

        let now = Instant::now();
        Self {
            camera,
            camera_binding,
            material,
            instances,
            templates,
            cube_vertices,
            cube_indices,
            buckets,
            particle_pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            particle_instances,
            emitters,
            visuals,
            started: now,
            last_frame: now,
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

        // Wall-clock orbit and dt — particle integration runs on wall-clock
        // so emitters keep ticking even when the sim is paused.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 7.5;
        let angle = t * 0.35;
        self.camera.position = Vec3::new(angle.sin() * radius, 2.5, angle.cos() * radius);
        self.camera.target = Vec3::ZERO;
        self.camera_binding.write(&renderer.queue, &self.camera);

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // Extract: walk sim, look up template, fan out into mesh + emitter
        // parts. Mesh parts go into instance buckets; emitter parts are
        // declared on the reconciler (which lazily creates or refreshes the
        // live emitter at this frame's world transform).
        self.buckets.begin_frame();
        self.emitters.begin_frame();
        for (zone_id, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&rid) = zone.components().get::<RenderId>(id) else {
                    continue;
                };
                let Some(template) = self.templates.get(rid) else {
                    continue;
                };
                let object_xform = Mat4::from_rotation_translation(obj.rotation, obj.position);
                let world_ref = WorldObjectRef { zone: zone_id, id };
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
                for part in template.emitter_parts() {
                    let world = object_xform * part.local_transform;
                    self.emitters
                        .declare(world_ref, part.slot, part.template, world);
                }
            }
        }

        // Tick particles, then turn each into a billboard instance with the
        // template's size/colour curves applied.
        self.emitters.integrate(&sim.zones, dt);
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

        self.buckets.upload(&renderer.queue);
        self.particle_instances.upload(&renderer.queue);

        // Opaque mesh pass: one indexed-instanced call per non-empty bucket.
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

        // Particle pass: alpha-blended billboards.
        pass.set_pipeline(&self.particle_pipeline);
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        for (_emitter_id, instance_buffer, count) in self.particle_instances.iter_filled() {
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..count);
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

fn build_particle_pipeline(
    device: &wgpu::Device,
    camera_layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle pipeline layout"),
        bind_group_layouts: &[Some(camera_layout)],
        ..Default::default()
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[
                // Slot 0: per-vertex quad corner.
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadCorner>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                },
                // Slot 1: per-instance ParticleInstance.
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
            // don't write depth (so overlapping particles blend correctly).
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
    currawong::run::<Demo>(Game::new());
}
