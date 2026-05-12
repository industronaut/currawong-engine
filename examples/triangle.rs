//! Draws a coloured triangle into the engine-provided render pass.

use currawong::{Renderer, View, ViewConfig, wgpu};

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
    );
    var colours = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.25, 0.25),
        vec3<f32>(0.25, 1.0, 0.4),
        vec3<f32>(0.3, 0.5, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.colour = colours[idx];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
"#;

struct Triangle {
    pipeline: wgpu::RenderPipeline,
}

impl View for Triangle {
    type Sim = ();

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — triangle",
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        let shader = renderer
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("triangle shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        let layout = renderer
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("triangle pipeline layout"),
                bind_group_layouts: &[],
                ..Default::default()
            });

        let pipeline = renderer
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("triangle pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
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

        Self { pipeline }
    }

    fn render(&mut self, _: &(), _: f32, _: &Renderer, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

fn main() {
    currawong::run::<Triangle>(());
}
