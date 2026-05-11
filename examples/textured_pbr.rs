//! Textured PBR cubes lit by a sun whose direction is driven by the sim's
//! [`SimEnvironment::time_of_day`]. The sun visibly crosses the sky as the
//! sim ticks — the cleanest possible demonstration of sim → view extraction
//! producing the lighting.
//!
//! Five cubes share one albedo texture (a procedural checkerboard) and
//! differ only in `(metallic, roughness)`:
//!
//! ```text
//! left ←-------------------------------------------------→ right
//!  matte    rough metal   semi-rough metal   shiny metal   glossy plastic
//! ```
//!
//! Controls:
//! - Space      — toggle pause.
//! - `1..=4`    — sim speed: 1×, 4×, 16×, 64× (default 1×).
//! - Esc        — quit.

use std::time::Duration;
use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, InstanceBuckets, MaterialInstanceRegistry,
    PbrInstanceAttribs, PbrMaterial, PbrMaterialInstance, PbrMaterialParams, PosNormalUv, Renderer,
    SamplerKind, SamplerRegistry, SimEnvironment, Simulation, Texture, View, ViewEnvironment,
    WorldObject, Zone, ZoneId, Zones, sun_direction_for, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ----------------------------------------------------------------

/// Which material instance a given cube renders with. Sim-side tag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MaterialId {
    Matte,
    RoughMetal,
    SemiRoughMetal,
    ShinyMetal,
    GlossyPlastic,
}

const ALL_MATERIALS: [MaterialId; 5] = [
    MaterialId::Matte,
    MaterialId::RoughMetal,
    MaterialId::SemiRoughMetal,
    MaterialId::ShinyMetal,
    MaterialId::GlossyPlastic,
];

struct Game {
    zones: Zones,
    zone: ZoneId,
    env: SimEnvironment,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");

        // Five cubes along the X axis, evenly spaced.
        for (i, &mat) in ALL_MATERIALS.iter().enumerate() {
            let x = (i as f32 - 2.0) * 1.5;
            let id = zone.insert(WorldObject {
                position: Vec3::new(x, 0.0, 0.5),
                rotation: Quat::IDENTITY,
            });
            zone.components_mut().insert(id, mat);
        }

        let mut env = SimEnvironment::new();
        // Compress the day so the sun crosses the sky quickly at 1× speed.
        env.seconds_per_day = 30.0;
        // Start an hour after sunrise so the cubes are lit at startup.
        env.time_of_day = 0.30;

        Self {
            zones,
            zone: zone_id,
            env,
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        self.env.advance(dt.as_secs_f32());
    }
}

// --- Mesh data ----------------------------------------------------------

/// Build a 1×1×1 cube as 24 [`PosNormalUv`] vertices (4 per face) + 36
/// indices. Per-face normals (so each face is flat-shaded) and UVs that
/// span 0..1 across the face.
fn cube_mesh() -> (Vec<PosNormalUv>, Vec<u16>) {
    // Six faces. For each: (normal, four corner positions ccw seen from outside).
    let h = 0.5;
    let faces: [(Vec3, [Vec3; 4]); 6] = [
        (
            // +X
            Vec3::X,
            [
                Vec3::new(h, -h, -h),
                Vec3::new(h, h, -h),
                Vec3::new(h, h, h),
                Vec3::new(h, -h, h),
            ],
        ),
        (
            // -X
            -Vec3::X,
            [
                Vec3::new(-h, h, -h),
                Vec3::new(-h, -h, -h),
                Vec3::new(-h, -h, h),
                Vec3::new(-h, h, h),
            ],
        ),
        (
            // +Y
            Vec3::Y,
            [
                Vec3::new(h, h, -h),
                Vec3::new(-h, h, -h),
                Vec3::new(-h, h, h),
                Vec3::new(h, h, h),
            ],
        ),
        (
            // -Y
            -Vec3::Y,
            [
                Vec3::new(-h, -h, -h),
                Vec3::new(h, -h, -h),
                Vec3::new(h, -h, h),
                Vec3::new(-h, -h, h),
            ],
        ),
        (
            // +Z (top)
            Vec3::Z,
            [
                Vec3::new(-h, -h, h),
                Vec3::new(h, -h, h),
                Vec3::new(h, h, h),
                Vec3::new(-h, h, h),
            ],
        ),
        (
            // -Z (bottom)
            -Vec3::Z,
            [
                Vec3::new(-h, h, -h),
                Vec3::new(h, h, -h),
                Vec3::new(h, -h, -h),
                Vec3::new(-h, -h, -h),
            ],
        ),
    ];
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (face_idx, (normal, corners)) in faces.iter().enumerate() {
        let base = (face_idx * 4) as u16;
        for (corner, uv) in corners.iter().zip(uvs.iter()) {
            verts.push(PosNormalUv {
                position: corner.to_array(),
                normal: normal.to_array(),
                uv: *uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// Procedural 64×64 RGBA8 checkerboard. Two tones of warm grey so the PBR
/// lighting reads against a neutral surface.
fn checkerboard_rgba(width: u32, height: u32, cells: u32) -> Vec<u8> {
    let cell_w = width / cells;
    let cell_h = height / cells;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let cx = x / cell_w;
            let cy = y / cell_h;
            let dark = (cx + cy) & 1 == 0;
            let (r, g, b) = if dark {
                (170u8, 165, 158)
            } else {
                (220u8, 215, 205)
            };
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 16;

struct TexturedPbr {
    camera: Camera,
    camera_binding: CameraBinding,
    material: PbrMaterial,
    instances: MaterialInstanceRegistry<PbrMaterialInstance, MaterialId>,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    cube_index_count: u32,
    buckets: InstanceBuckets<MaterialId, PbrInstanceAttribs>,
    started: Instant,
}

impl View for TexturedPbr {
    type Sim = Game;

    fn init(renderer: &Renderer) -> Self {
        use wgpu::util::DeviceExt;

        let device = &renderer.device;
        let camera = Camera {
            position: Vec3::new(0.0, -8.0, 3.0),
            target: Vec3::new(0.0, 0.0, 0.5),
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(device);
        let samplers = SamplerRegistry::new(device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());

        // One albedo texture shared by all five instances.
        let checker = checkerboard_rgba(64, 64, 8);
        let albedo = Texture::from_rgba8(renderer, "checkerboard", 64, 64, &checker, true);

        let make = |metallic: f32, roughness: f32| {
            material.create_instance(
                renderer,
                &samplers,
                PbrMaterialParams {
                    albedo: &albedo,
                    sampler: SamplerKind::LinearRepeat,
                    albedo_factor: Vec4::ONE,
                    metallic,
                    roughness,
                },
            )
        };
        let mut instances = MaterialInstanceRegistry::new();
        instances.register(MaterialId::Matte, make(0.0, 0.95));
        instances.register(MaterialId::RoughMetal, make(1.0, 0.75));
        instances.register(MaterialId::SemiRoughMetal, make(1.0, 0.45));
        instances.register(MaterialId::ShinyMetal, make(1.0, 0.15));
        instances.register(MaterialId::GlossyPlastic, make(0.0, 0.18));

        let (verts, indices) = cube_mesh();
        let cube_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut buckets = InstanceBuckets::<MaterialId, PbrInstanceAttribs>::new(
            "pbr instance attribs",
            MAX_INSTANCES,
        );
        for &mat in ALL_MATERIALS.iter() {
            buckets.register(device, mat);
        }

        Self {
            camera,
            camera_binding,
            material,
            instances,
            cube_vertices,
            cube_indices,
            cube_index_count: indices.len() as u32,
            buckets,
            started: Instant::now(),
        }
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.zone)
    }

    fn extract_environment(&self, sim: &Game, _zone: ZoneId) -> ViewEnvironment {
        let sun = sun_direction_for(sim.env.time_of_day);
        // Sun intensity rolls off as the sun dips toward and below the
        // horizon (sun.z is the "above-horizon" component in our Z-up frame).
        // Quadratic falloff feels less abrupt than linear.
        let above_horizon = sun.z.max(0.0);
        let intensity = above_horizon * above_horizon * 3.0;
        // Warm sunrise/sunset by tinting toward orange when low; cool when
        // high. Very rough but visibly conveys time of day.
        let warm = Vec3::new(1.0, 0.55, 0.30);
        let cool = Vec3::new(1.0, 0.97, 0.92);
        let warmth = (1.0 - above_horizon).clamp(0.0, 1.0).powf(2.0);
        let tint = cool.lerp(warm, warmth);
        let sun_color = tint * intensity;

        // Sky/ambient also dim at night.
        let day = above_horizon.clamp(0.0, 1.0).powf(0.5);
        let sky = Vec3::new(0.45, 0.65, 0.95).lerp(Vec3::new(0.02, 0.03, 0.06), 1.0 - day);
        let ambient = Vec3::splat(0.05).lerp(Vec3::new(0.35, 0.40, 0.50), day);

        ViewEnvironment {
            sun_direction: sun,
            sun_color,
            ambient,
            sky_color: sky,
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

        // Wall-clock camera orbit so the cubes are visible from changing
        // angles even when the sim is paused.
        let t = self.started.elapsed().as_secs_f32();
        let radius = 9.0;
        let angle = t * 0.25;
        self.camera.position = Vec3::new(angle.sin() * radius, -angle.cos() * radius, 3.5);
        self.camera.target = Vec3::new(0.0, 0.0, 0.5);
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Slow yaw on the cubes themselves so the lighting varies per-face.
        let cube_yaw = Quat::from_rotation_z(t * 0.4);

        self.buckets.begin_frame();
        for (_, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&mat) = zone.components().get::<MaterialId>(id) else {
                    continue;
                };
                let model = Mat4::from_rotation_translation(obj.rotation * cube_yaw, obj.position);
                self.buckets
                    .push(mat, PbrInstanceAttribs::new(model, Vec4::ONE));
            }
        }
        self.buckets.upload(&renderer.queue);

        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        pass.set_vertex_buffer(0, self.cube_vertices.slice(..));
        pass.set_index_buffer(self.cube_indices.slice(..), wgpu::IndexFormat::Uint16);
        for (mat, instance_buffer, count) in self.buckets.iter_filled() {
            let Some(instance) = self.instances.get(mat) else {
                continue;
            };
            pass.set_bind_group(2, instance.bind_group(), &[]);
            pass.set_vertex_buffer(1, instance_buffer.slice(..));
            pass.draw_indexed(0..self.cube_index_count, 0, 0..count);
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
            KeyCode::Space => {
                let new_speed = if ctx.clock.is_paused() { 1.0 } else { 0.0 };
                ctx.clock.set_speed(new_speed);
            }
            KeyCode::Digit1 => ctx.clock.set_speed(1.0),
            KeyCode::Digit2 => ctx.clock.set_speed(4.0),
            KeyCode::Digit3 => ctx.clock.set_speed(16.0),
            KeyCode::Digit4 => ctx.clock.set_speed(64.0),
            _ => {}
        }
    }

    fn title() -> &'static str {
        "currawong — textured PBR cubes under a moving sun"
    }

    fn clear_colour() -> wgpu::Color {
        // Daylight blue. Sky/ambient inside the shader still varies with
        // time of day; the clear is currently static (clear_colour is an
        // associated method, not bound to per-frame state).
        wgpu::Color {
            r: 0.45,
            g: 0.65,
            b: 0.95,
            a: 1.0,
        }
    }

    fn depth_format() -> Option<wgpu::TextureFormat> {
        Some(DEPTH_FORMAT)
    }
}

fn main() {
    currawong::run::<TexturedPbr>(Game::new());
}
