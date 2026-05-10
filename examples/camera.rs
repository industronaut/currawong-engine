//! Sim-driven rendering: three WorldObjects bobbing on Y, viewed through a Camera.
//!
//! Demonstrates the sim/view extract path: `Game` ticks WorldObjects in its
//! `Zones`; `SimDriven` reads their positions each frame, uploads them as an
//! instance buffer, and renders one triangle per object through a Camera's
//! view-projection matrix.
//!
//! Controls: 0 pause, 1/2/3 set sim speed, Esc to quit.

use std::time::Duration;

use currawong::glam::{Quat, Vec3};
use currawong::{
    Camera, EngineCtx, Renderer, Simulation, View, WorldObject, Zone, ZoneId, Zones, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Simulation ----------------------------------------------------------

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
        for x in [-2.0, 0.0, 2.0] {
            zone.insert(WorldObject {
                position: Vec3::new(x, 0.0, 0.0),
                rotation: Quat::IDENTITY,
            });
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
        // Bob each object on Y with a phase offset, so motion is visibly tied
        // to sim ticks rather than render frames. Watching with speed 0.5x vs
        // 2x makes the sim/view decoupling obvious.
        let zone = self.zones.get_mut(self.main_zone).expect("main zone");
        for (i, (_, obj)) in zone.iter_mut().enumerate() {
            let phase = i as f32 * 1.5;
            obj.position.y = (t * 2.0 + phase).sin() * 0.6;
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
    @location(0) instance_pos: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var local = array<vec3<f32>, 3>(
        vec3<f32>( 0.0,  0.4, 0.0),
        vec3<f32>(-0.4, -0.4, 0.0),
        vec3<f32>( 0.4, -0.4, 0.0),
    );
    var colours = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.25, 0.25),
        vec3<f32>(0.25, 1.0, 0.4),
        vec3<f32>(0.3, 0.5, 1.0),
    );
    let world = local[in.vidx] + in.instance_pos;
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.colour = colours[in.vidx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
"#;

const MAX_INSTANCES: u64 = 256;
const INSTANCE_SIZE: u64 = std::mem::size_of::<[f32; 3]>() as u64;

struct SimDriven {
    camera: Camera,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_scratch: Vec<[f32; 3]>,
}

impl View for SimDriven {
    type Sim = Game;

    fn init(renderer: &Renderer) -> Self {
        let device = &renderer.device;

        let camera = Camera::default();

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<[f32; 16]>() as u64,
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
            bind_group_layouts: &[Some(&camera_bgl)],
            ..Default::default()
        });

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
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera,
            pipeline,
            camera_buffer,
            camera_bind_group,
            instance_buffer,
            instance_scratch: Vec::with_capacity(MAX_INSTANCES as usize),
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

        // Upload the view-projection matrix.
        let view_proj = self.camera.view_proj();
        renderer
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&view_proj));

        // Extract WorldObject positions across all zones into the instance buffer.
        self.instance_scratch.clear();
        for (_, zone) in sim.zones.iter() {
            for (_, obj) in zone.iter() {
                self.instance_scratch.push(obj.position.to_array());
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
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
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

    fn title() -> &'static str {
        "currawong — camera demo"
    }
}

fn main() {
    currawong::run::<SimDriven>(Game::new());
}
