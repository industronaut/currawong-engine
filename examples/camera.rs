//! Sim-driven rendering: three WorldObjects bobbing on Z, viewed through an
//! orbiting Camera with depth-tested rendering.
//!
//! Demonstrates the sim/view extract path: `Game` ticks WorldObjects in its
//! `Zones`; `SimDriven` reads their positions each frame, uploads them as an
//! instance buffer, and renders one triangle per object through a Camera's
//! view-projection matrix. Camera orbit runs off a wall-clock Instant, so it
//! keeps moving even at sim speed 0 — visible proof the view is independent.
//!
//! Controls: 0 pause, 1/2/3 set sim speed, Esc to quit.

use std::time::{Duration, Instant};

use currawong::glam::{Mat4, Quat, Vec3};
use currawong::{
    Camera, CameraBinding, EngineCtx, Renderer, Simulation, View, ViewConfig, WorldObject, Zone,
    ZoneId, Zones, mat4_instance_attributes, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Simulation ----------------------------------------------------------

struct Game {
    zones: Zones,
    main_zone: ZoneId,
    elapsed: Duration,
}

/// Per-object bobbing parameters, attached as a component. Demonstrates the
/// component pattern: per-object data the engine doesn't know about, looked
/// up sparsely by [`WorldObjectId`].
struct Bobber {
    phase: f32,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let main_zone = zones.insert(Zone::new());
        let zone = zones.get_mut(main_zone).expect("just inserted");
        // Stagger in both X and Y so the orbiting camera sees their depth
        // ordering swap as it goes around — without depth testing, the wrong
        // triangle would be on top from half the angles.
        for (i, (x, y)) in [(-2.0, -1.0), (0.0, 0.0), (2.0, 1.0)].iter().enumerate() {
            let id = zone.insert(WorldObject {
                position: Vec3::new(*x, *y, 0.0),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(
                id,
                Bobber {
                    phase: i as f32 * 1.5,
                },
            );
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
        // Iterate the Bobber component and write back to each object's
        // transform. `split_mut` hands out independent borrows of the object
        // map and the component registry so we can do both in one pass.
        let zone = self.zones.get_mut(self.main_zone).expect("main zone");
        let (objects, components) = zone.split_mut();
        for (id, bobber) in components.iter::<Bobber>() {
            if let Some(obj) = objects.get_mut(id) {
                obj.position.z = (t * 2.0 + bobber.phase).sin() * 0.6;
                // Spin around the world up axis (Z) so the rotation is
                // visible as a yaw from the orbiting camera — different rates
                // per object via phase.
                obj.rotation = Quat::from_rotation_z(t * 0.8 + bobber.phase);
            }
        }
    }
}

// --- View ----------------------------------------------------------------

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @builtin(vertex_index) vidx: u32,
    // Per-instance model matrix, supplied as four columns. WGSL doesn't allow
    // a mat4x4 vertex attribute directly, so we receive 4 vec4s and rebuild it.
    @location(0) m0: vec4<f32>,
    @location(1) m1: vec4<f32>,
    @location(2) m2: vec4<f32>,
    @location(3) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Z-up local space: triangle stands upright in the XZ plane so the
    // orbiting camera sees it face-on rather than edge-on.
    var local = array<vec3<f32>, 3>(
        vec3<f32>( 0.0, 0.0,  0.4),
        vec3<f32>(-0.4, 0.0, -0.4),
        vec3<f32>( 0.4, 0.0, -0.4),
    );
    var colours = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.25, 0.25),
        vec3<f32>(0.25, 1.0, 0.4),
        vec3<f32>(0.3, 0.5, 1.0),
    );
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world = model * vec4<f32>(local[in.vidx], 1.0);
    var out: VsOut;
    out.clip = camera.view_proj * world;
    out.colour = colours[in.vidx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
"#;

const MAX_INSTANCES: u64 = 256;
const INSTANCE_SIZE: u64 = std::mem::size_of::<Mat4>() as u64;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct SimDriven {
    camera: Camera,
    camera_binding: CameraBinding,
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    instance_scratch: Vec<Mat4>,
    started: Instant,
}

impl View for SimDriven {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — camera demo",
        depth_format: Some(DEPTH_FORMAT),
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let camera = Camera::default();
        let camera_binding = CameraBinding::new(device);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance buffer"),
            size: MAX_INSTANCES * INSTANCE_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera demo shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("camera demo layout"),
            bind_group_layouts: &[Some(camera_binding.layout())],
            ..Default::default()
        });

        let mat4_attrs = mat4_instance_attributes(0);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("camera demo pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: INSTANCE_SIZE,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &mat4_attrs,
                }],
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

        Self {
            camera,
            camera_binding,
            pipeline,
            instance_buffer,
            instance_scratch: Vec::with_capacity(MAX_INSTANCES as usize),
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
        // Refresh aspect — the user may have resized since the last frame.
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }

        // Orbit on a wall-clock timer, deliberately decoupled from sim time:
        // pausing the sim (speed 0) leaves the camera moving, which makes the
        // independence of view from sim observable.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 5.5;
        let angle = t * 0.4;
        self.camera.position = Vec3::new(angle.sin() * radius, angle.cos() * radius, 2.0);

        // Upload the camera uniform (view-proj + billboard basis; this view's
        // shader only reads view_proj but the engine helper writes both).
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Extract per-object model matrices across all zones.
        self.instance_scratch.clear();
        for (_, zone) in sim.zones.iter() {
            for (_, obj) in zone.iter() {
                self.instance_scratch
                    .push(Mat4::from_rotation_translation(obj.rotation, obj.position));
                if self.instance_scratch.len() == MAX_INSTANCES as usize {
                    break;
                }
            }
        }
        let count = self.instance_scratch.len() as u32;
        if count > 0 {
            renderer.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_scratch),
            );
        }

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        pass.draw(0..3, 0..count);
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
    currawong::run::<SimDriven>(Game::new());
}
