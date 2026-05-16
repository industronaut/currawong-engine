//! Textured PBR cubes lit by a sun whose direction is driven by the sim's
//! [`SimEnvironment::time_of_day`]. The sun visibly crosses the sky as the
//! sim ticks — the cleanest possible demonstration of sim → view extraction
//! producing the lighting.
//!
//! Five cubes share one albedo texture (loaded from
//! `assets/checker_albedo.png`) and differ only in `(metallic, roughness)`:
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
//!
//! Build with `--features egui` to enable a small debug panel: FPS, sim
//! clock readout, pause/speed widgets, and a time-of-day slider you can
//! drag to move the sun directly. Without the feature the example still
//! runs; it just lacks the overlay.

#[cfg(feature = "egui")]
use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

#[cfg(feature = "egui")]
use currawong::egui;
use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Camera, CameraBinding, EngineCtx, InstanceBuckets, MaterialInstanceRegistry,
    MeshInstanceAttribs, PbrMaterial, PbrMaterialInstance, PbrMaterialParams, PrimitiveMesh,
    Renderer, SamplerKind, SamplerRegistry, SimEnvironment, Simulation, Texture, TextureColorSpace,
    View, ViewConfig, ViewEnvironment, WorldTransform, Zone, ZoneId, Zones, sun_direction_for,
    wgpu, winit,
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
            let id = zone.insert(WorldTransform {
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
    buckets: InstanceBuckets<MaterialId, MeshInstanceAttribs>,
    started: Instant,
    #[cfg(feature = "egui")]
    frame_samples: VecDeque<f32>,
    #[cfg(feature = "egui")]
    last_frame: Instant,
}

impl View for TexturedPbr {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — textured PBR cubes under a moving sun",
        // Daylight blue. Sky/ambient inside the shader still varies with
        // time of day; the clear is currently static.
        clear_colour: wgpu::Color {
            r: 0.45,
            g: 0.65,
            b: 0.95,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
    };

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

        // One albedo texture shared by all five instances. Resolved
        // relative to the crate root so the example runs from any cwd.
        let albedo_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/checker_albedo.png");
        let albedo = Texture::from_path(renderer, &albedo_path, TextureColorSpace::Srgb)
            .expect("load assets/checker_albedo.png");

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

        let mesh = PrimitiveMesh::cube(Vec3::ONE);
        let cube_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut buckets = InstanceBuckets::<MaterialId, MeshInstanceAttribs>::new(
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
            cube_index_count: mesh.index_count(),
            buckets,
            started: Instant::now(),
            #[cfg(feature = "egui")]
            frame_samples: VecDeque::with_capacity(120),
            #[cfg(feature = "egui")]
            last_frame: Instant::now(),
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
        for (zone_id, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let Some(&mat) = zone.components().get::<MaterialId>(id) else {
                    continue;
                };
                let model = Mat4::from_rotation_translation(obj.rotation * cube_yaw, obj.position);
                // Reserve a per-cube hit ID so the GPU readback can resolve
                // a cursor over any of these cubes back to its sim
                // `WorldObjectId` (#56 PR 3).
                let hit_id = renderer.reserve_object(zone_id, id);
                self.buckets.push(
                    mat,
                    MeshInstanceAttribs::new(model, Vec4::ONE).with_hit_id(hit_id),
                );
            }
        }
        self.buckets.upload(&renderer.queue);

        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        pass.set_vertex_buffer(0, self.cube_vertices.slice(..));
        pass.set_index_buffer(self.cube_indices.slice(..), wgpu::IndexFormat::Uint32);
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

    #[cfg(feature = "egui")]
    fn ui(&mut self, sim: &mut Game, ctx: &mut EngineCtx, egui_ctx: &egui::Context) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_samples.len() == 120 {
            self.frame_samples.pop_front();
        }
        self.frame_samples.push_back(dt);
        let avg_dt = self.frame_samples.iter().sum::<f32>() / self.frame_samples.len() as f32;
        let fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };

        egui::Window::new("debug")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(egui_ctx, |ui| {
                ui.label(format!("fps: {fps:5.1}  ({:.2} ms)", avg_dt * 1000.0));
                ui.separator();

                ui.label(format!("sim ticks: {}", ctx.clock.total_ticks()));
                let mut speed = ctx.clock.speed();
                ui.horizontal(|ui| {
                    let label = if ctx.clock.is_paused() {
                        "play"
                    } else {
                        "pause"
                    };
                    if ui.button(label).clicked() {
                        let new = if ctx.clock.is_paused() { 1.0 } else { 0.0 };
                        ctx.clock.set_speed(new);
                        speed = new;
                    }
                    ui.add(egui::Slider::new(&mut speed, 0.0..=64.0).text("speed"));
                });
                if (speed - ctx.clock.speed()).abs() > f32::EPSILON {
                    ctx.clock.set_speed(speed);
                }
                ui.separator();

                ui.label(format!("day: {}", sim.env.day));
                ui.add(
                    egui::Slider::new(&mut sim.env.time_of_day, 0.0..=1.0)
                        .text("time of day")
                        .custom_formatter(|v, _| format_time_of_day(v as f32)),
                );
                let sun = sun_direction_for(sim.env.time_of_day);
                ui.label(format!(
                    "sun dir: ({:+.2}, {:+.2}, {:+.2})",
                    sun.x, sun.y, sun.z
                ));
                ui.label(if sun.z > 0.0 {
                    "above horizon"
                } else {
                    "below horizon"
                });
                ui.separator();

                if ui.button("quit").clicked() {
                    ctx.event_loop.exit();
                }
            });
    }
}

#[cfg(feature = "egui")]
fn format_time_of_day(t: f32) -> String {
    let total_minutes = (t.rem_euclid(1.0) * 24.0 * 60.0) as i32;
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    format!("{hours:02}:{minutes:02}")
}

fn main() {
    currawong::run::<TexturedPbr>(Game::new());
}
