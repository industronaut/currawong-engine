//! Multi-mesh bucketed instancing via the declarative-extract pattern.
//!
//! Three sim "kinds" (Building, Tree, Rock) — pure semantic state, no visual
//! concepts. The View decides what each kind looks like (Cube, Tetra, Plane),
//! walks the sim each frame, pushes per-object model matrices into the
//! matching mesh's instance batch, and issues one instanced draw per mesh.
//!
//! Architectural shape:
//! - `extract_visuals(zone, id, obj, push)` is the bridge. It reads sim
//!   state and pushes `(MeshId, Mat4)` for each visible part. Right now each
//!   sim object yields one part; the same shape extends to N parts (composite
//!   props, articulated rigs) and later to particle emitter attachments.
//! - The View owns the mesh table, the pipeline, and the camera. The
//!   per-mesh instance scratch + GPU buffers are managed by the engine's
//!   `InstanceBuckets<MeshId, Mat4>` helper. Sim is renderer-ignorant.
//!
//! Controls: 0 pause, 1/2/3 set sim speed, Esc to quit.
//! Sim ticks drive only the trees' sway rotation — buildings and rocks are
//! static. Camera orbits on wall-clock so it keeps moving when paused.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Quat, Vec3};
use currawong::{
    Camera, EngineCtx, InstanceBuckets, Renderer, Simulation, View, ViewConfig, WorldObjectId,
    WorldTransform, Zone, ZoneId, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Simulation ----------------------------------------------------------

/// Semantic kind of a world object. *Sim* concept — the View decides what
/// each kind looks like. Adding a new kind on the sim side does not require
/// any visual code; adding a new visual representation does not require any
/// sim change.
#[derive(Clone, Copy)]
enum Kind {
    Building,
    Tree,
    Rock,
}

/// Per-tree sway phase. Sparse component — only attached to objects whose
/// kind is `Tree`. Demonstrates that gameplay-relevant state lives in
/// components, not on every WorldTransform.
struct TreeSway {
    phase: f32,
}

struct Game {
    zones: Zones,
    main_zone: ZoneId,
    elapsed: Duration,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let main_zone = zones.insert(Zone::new());
        let zone = zones.get_mut(main_zone).expect("just inserted");

        // Buildings on the +X side.
        for x in [1.6, 3.0, 4.4] {
            let id = zone.insert(WorldTransform {
                position: Vec3::new(x, 0.0, 0.5),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, Kind::Building);
        }

        // Trees on the -X side. Stagger phase so they don't sway in lock-step.
        for (i, x) in [-1.6, -3.0, -4.4].iter().enumerate() {
            let id = zone.insert(WorldTransform {
                position: Vec3::new(*x, 0.0, 0.4),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, Kind::Tree);
            zone.components_mut().insert(
                id,
                TreeSway {
                    phase: i as f32 * 1.3,
                },
            );
        }

        // Rocks scattered along Y, sitting low.
        for y in [-2.2, -0.6, 0.6, 2.2] {
            let id = zone.insert(WorldTransform {
                position: Vec3::new(0.0, y, -0.45),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, Kind::Rock);
        }

        Self {
            zones,
            main_zone,
            elapsed: Duration::ZERO,
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        self.elapsed += dt;
        let t = self.elapsed.as_secs_f32();
        // Only trees sway. `split_mut` lets us iterate the TreeSway
        // component map and mutate the matching objects in one pass.
        let zone = self.zones.get_mut(self.main_zone).expect("main zone");
        let (mut objects, components) = zone.split_mut();
        for (id, sway) in components.iter::<TreeSway>() {
            if let Some(obj) = objects.get_mut(id) {
                let angle = (t * 1.4 + sway.phase).sin() * 0.22;
                // Tilt around the world Y axis so the tree leans in the XZ
                // plane (a perpendicular sway for trees laid out along X).
                obj.rotation = Quat::from_rotation_y(angle);
            }
        }
    }
}

// --- Mesh data (CPU-side, baked into GPU buffers at View::init) ----------

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    pos: [f32; 3],
    col: [f32; 3],
}

// Trivially safe: #[repr(C)] struct of Pod fields, no padding, no Drop.
unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

const fn v(p: [f32; 3], c: [f32; 3]) -> Vertex {
    Vertex { pos: p, col: c }
}

// Unit cube centred on origin. Per-corner colour shows the orientation of
// each vertex without requiring lighting — adjacent corners differ by axis.
// Local space is Z-up: top face is +Z, side faces are ±X / ±Y.
#[rustfmt::skip]
const CUBE_VERTS: &[Vertex] = &[
    v([-0.5, -0.5, -0.5], [0.25, 0.40, 0.70]),
    v([ 0.5, -0.5, -0.5], [0.65, 0.40, 0.70]),
    v([ 0.5,  0.5, -0.5], [0.65, 0.75, 0.70]),
    v([-0.5,  0.5, -0.5], [0.25, 0.75, 0.70]),
    v([-0.5, -0.5,  0.5], [0.25, 0.40, 0.95]),
    v([ 0.5, -0.5,  0.5], [0.65, 0.40, 0.95]),
    v([ 0.5,  0.5,  0.5], [0.65, 0.75, 0.95]),
    v([-0.5,  0.5,  0.5], [0.25, 0.75, 0.95]),
];
#[rustfmt::skip]
const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, // -Z (bottom)
    4, 6, 5, 4, 7, 6, // +Z (top)
    0, 3, 7, 0, 7, 4, // -X (left)
    1, 5, 6, 1, 6, 2, // +X (right)
    3, 2, 6, 3, 6, 7, // +Y (back)
    0, 4, 5, 0, 5, 1, // -Y (front)
];

// Regular tetrahedron, point up along local +Z.
#[rustfmt::skip]
const TETRA_VERTS: &[Vertex] = &[
    v([ 0.0,  0.0,  0.65], [0.30, 0.90, 0.40]),
    v([-0.55,  0.55, -0.30], [0.20, 0.70, 0.30]),
    v([ 0.55,  0.55, -0.30], [0.45, 0.85, 0.30]),
    v([ 0.0, -0.55, -0.30], [0.30, 0.65, 0.20]),
];
#[rustfmt::skip]
const TETRA_INDICES: &[u16] = &[
    0, 1, 2, // back face
    0, 2, 3, // right face
    0, 3, 1, // left face
    1, 3, 2, // bottom face
];

// Quad in the XY plane (Z-up local space). Used flat as ground tiles.
#[rustfmt::skip]
const PLANE_VERTS: &[Vertex] = &[
    v([-0.6, -0.6, 0.0], [0.55, 0.50, 0.45]),
    v([ 0.6, -0.6, 0.0], [0.60, 0.55, 0.50]),
    v([ 0.6,  0.6, 0.0], [0.55, 0.50, 0.45]),
    v([-0.6,  0.6, 0.0], [0.60, 0.55, 0.50]),
];
#[rustfmt::skip]
const PLANE_INDICES: &[u16] = &[
    0, 1, 2,
    0, 2, 3,
];

// --- View: meshes + bucketed instance batches ----------------------------

/// Identifies a mesh kind. The View keeps a `HashMap<MeshId, Mesh>` of GPU
/// data and uses the same key as the bucket key for `InstanceBuckets`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MeshId {
    Cube,
    Tetra,
    Plane,
}

/// GPU-resident mesh data. Uploaded once at init.
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

/// Declarative per-frame visual extraction.
///
/// Reads sim state and pushes one or more `(MeshId, Mat4)` pairs per visible
/// part. Currently each sim object yields exactly one part, but the same
/// signature handles composite visuals (push multiple times) and will extend
/// to particle emitter attachments by widening the leaf type.
fn extract_visuals(
    zone: &Zone,
    id: WorldObjectId,
    obj: &WorldTransform,
    push: &mut dyn FnMut(MeshId, Mat4),
) {
    let Some(kind) = zone.components().get::<Kind>(id) else {
        return;
    };
    let mesh = match kind {
        Kind::Building => MeshId::Cube,
        Kind::Tree => MeshId::Tetra,
        Kind::Rock => MeshId::Plane,
    };
    let model = Mat4::from_rotation_translation(obj.rotation, obj.position);
    push(mesh, model);
}

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) col: vec3<f32>,
    // Per-instance model matrix as four vec4 columns.
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

const MAX_INSTANCES_PER_MESH: u32 = 256;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct Instances {
    camera: Camera,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    meshes: HashMap<MeshId, Mesh>,
    instances: InstanceBuckets<MeshId, Mat4>,
    started: Instant,
}

impl View for Instances {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — multi-mesh instances",
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let camera = Camera::default();

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<Mat4>() as u64,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instances shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instances layout"),
            bind_group_layouts: &[Some(&camera_bgl)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instances pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Slot 0: per-vertex (pos + colour). Same layout for all meshes.
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
                    // Slot 1: per-instance model matrix as four vec4 columns.
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
                    format: renderer.surface_format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // No back-face culling: the hand-rolled mesh windings in this
            // example aren't all CCW-outward. A real renderer would commit
            // to consistent winding and turn culling on for the perf win.
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
        });

        let mut meshes = HashMap::new();
        meshes.insert(
            MeshId::Cube,
            Mesh::new(device, "cube", CUBE_VERTS, CUBE_INDICES),
        );
        meshes.insert(
            MeshId::Tetra,
            Mesh::new(device, "tetra", TETRA_VERTS, TETRA_INDICES),
        );
        meshes.insert(
            MeshId::Plane,
            Mesh::new(device, "plane", PLANE_VERTS, PLANE_INDICES),
        );

        let mut instances =
            InstanceBuckets::<MeshId, Mat4>::new("mesh instances", MAX_INSTANCES_PER_MESH);
        for &key in &[MeshId::Cube, MeshId::Tetra, MeshId::Plane] {
            instances.register(device, key);
        }

        Self {
            camera,
            pipeline,
            camera_buffer,
            camera_bind_group,
            meshes,
            instances,
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

        // Wall-clock camera orbit — keeps moving when sim is paused, making
        // the sim/view decoupling visible.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 8.5;
        let angle = t * 0.35;
        self.camera.position = Vec3::new(angle.sin() * radius, angle.cos() * radius, 3.5);

        let view_proj = self.camera.view_proj();
        renderer
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&view_proj));

        // --- Extract phase: clear, walk sim, push into per-mesh buckets.
        self.instances.begin_frame();
        for (_, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                extract_visuals(zone, id, obj, &mut |mesh, model| {
                    self.instances.push(mesh, model);
                });
            }
        }

        // --- Upload phase: one write per non-empty bucket.
        self.instances.upload(&renderer.queue);

        // --- Draw phase: one instanced draw per non-empty bucket.
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for (mesh_id, instance_buffer, count) in self.instances.iter_filled() {
            let mesh = &self.meshes[&mesh_id];
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..count);
        }
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
    currawong::run::<Instances>(Game::new());
}
