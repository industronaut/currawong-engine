//! Streaming asset loading, demonstrated.
//!
//! A single PBR cube whose albedo is streamed from the VFS through the
//! [`AssetServer`]. On startup the cube renders magenta (the asset server's
//! resident fallback) while the background loader thread is in flight,
//! then snaps to the real texture once the load lands. The transition is
//! logged to stdout so it's observable even when the natural streaming
//! window is shorter than a frame.
//!
//! ## Controls
//!
//! - `F` (hold) — force every handle to *look* `Loading` to consumers.
//!   The cube rebinds to the magenta fallback for as long as the key is
//!   held; release it to see the real texture rebind. This is the debug
//!   toggle on [`AssetServer::set_force_loading`] — exists so the fallback
//!   path renders on demand during normal dev, not just during real I/O
//!   hiccups.
//! - `Esc` — quit.
//!
//! ## What this proves
//!
//! - `Handle::peek` returns `Loading` until the background thread
//!   publishes, then `Ready` for the rest of the run. The cube's bind
//!   group rebuilds exactly once across that transition (the
//!   [`PbrMaterialInstance::refresh`] cache check).
//! - The fallback is bindable any time the handle isn't `Ready`, including
//!   the synthetic "forced loading" state. The render path never has
//!   "nothing" to bind.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{FsSource, Vfs, VfsPath};
use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    AssetServer, Camera, CameraBinding, EngineCtx, Handle, HandleState, InstanceBuckets,
    MaterialInstanceRegistry, MeshInstanceAttribs, PbrMaterial, PbrMaterialInstance,
    PbrMaterialParams, PrimitiveMesh, Renderer, SamplerKind, SamplerRegistry, Simulation, Texture,
    TextureColorSpace, View, ViewConfig, ViewEnvironment, WorldTransform, Zone, ZoneId, Zones,
    sun_direction_for, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ----------------------------------------------------------------

/// Minimal sim: one object at origin, plus a `SimEnvironment` so the sun
/// has a direction.
struct Game {
    zones: Zones,
    zone: ZoneId,
    time_of_day: f32,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");
        zone.insert(WorldTransform {
            position: Vec3::new(0.0, 0.0, 0.5),
            rotation: Quat::IDENTITY,
        });
        Self {
            zones,
            zone: zone_id,
            // Mid-morning sun — bright enough to read both the magenta
            // fallback and the checker texture clearly.
            time_of_day: 0.30,
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        // Sun drifts slowly across the sky; visible motion not required
        // for this example, but keeps the lighting alive.
        let day_seconds = 90.0;
        self.time_of_day = (self.time_of_day + dt.as_secs_f32() / day_seconds).rem_euclid(1.0);
    }
}

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_INSTANCES: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MatId {
    Cube,
}

struct AssetsView {
    camera: Camera,
    camera_binding: CameraBinding,
    material: PbrMaterial,
    samplers: SamplerRegistry,
    asset_server: AssetServer,
    instances: MaterialInstanceRegistry<PbrMaterialInstance, MatId>,
    cube_vertices: wgpu::Buffer,
    cube_indices: wgpu::Buffer,
    cube_index_count: u32,
    buckets: InstanceBuckets<MatId, MeshInstanceAttribs>,
    /// Snapshot of last frame's handle state — used to log transitions
    /// without spamming stdout every frame.
    last_logged_state: LoggedState,
    /// Cube's albedo handle — kept here so we can peek it from `update`
    /// for the transition log.
    albedo: Handle<Texture>,
    /// Whether we logged the "press F" hint yet.
    hint_logged: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LoggedState {
    Loading,
    Ready,
    Failed,
}

impl View for AssetsView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — streaming asset (hold F to force fallback)",
        clear_colour: wgpu::Color {
            r: 0.10,
            g: 0.12,
            b: 0.16,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
    };

    fn init(renderer: &Renderer) -> Self {
        use wgpu::util::DeviceExt;

        let device = &renderer.device;
        let camera = Camera {
            position: Vec3::new(2.5, -3.5, 2.5),
            target: Vec3::new(0.0, 0.0, 0.5),
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(device);
        let samplers = SamplerRegistry::new(device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());

        // Mount the project's `assets/` directory as the VFS base layer.
        // Real games would mount a sealed archive in ship builds and the
        // loose directory in dev; both paths produce a `Vfs` the asset
        // server doesn't have to distinguish.
        let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mut vfs = Vfs::new();
        vfs.mount(FsSource::new(assets_root));
        let asset_server = AssetServer::new(renderer, Arc::new(vfs));

        // Spawn the streaming load. Returns immediately; the handle starts
        // `Loading` and transitions to `Ready` (or `Failed`) once the
        // background thread finishes the VFS read + PNG decode + upload.
        let albedo: Handle<Texture> = asset_server.texture(
            VfsPath::new("checker_albedo.png").expect("valid VFS path"),
            TextureColorSpace::Srgb,
        );

        let cube_instance = material.create_instance(
            renderer,
            &samplers,
            &asset_server,
            PbrMaterialParams {
                albedo: albedo.clone(),
                sampler: SamplerKind::LinearRepeat,
                albedo_factor: Vec4::ONE,
                metallic: 0.0,
                roughness: 0.5,
            },
        );
        let mut instances = MaterialInstanceRegistry::new();
        instances.register(MatId::Cube, cube_instance);

        let mesh = PrimitiveMesh::cube(Vec3::ONE);
        let cube_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("assets cube vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("assets cube indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut buckets = InstanceBuckets::<MatId, MeshInstanceAttribs>::new(
            "assets instance attribs",
            MAX_INSTANCES,
        );
        buckets.register(device, MatId::Cube);

        Self {
            camera,
            camera_binding,
            material,
            samplers,
            asset_server,
            instances,
            cube_vertices,
            cube_indices,
            cube_index_count: mesh.index_count(),
            buckets,
            last_logged_state: LoggedState::Loading,
            albedo,
            hint_logged: false,
        }
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.zone)
    }

    fn extract_environment(&self, sim: &Game, _zone: ZoneId) -> ViewEnvironment {
        let sun = sun_direction_for(sim.time_of_day);
        ViewEnvironment {
            sun_direction: sun,
            sun_color: Vec3::splat(2.0),
            ambient: Vec3::new(0.25, 0.27, 0.32),
            sky_color: Vec3::new(0.45, 0.55, 0.75),
        }
    }

    fn update(&mut self, _sim: &Game, _ctx: &mut EngineCtx, _dt: Duration) {
        if !self.hint_logged {
            println!("assets demo: hold F to force the magenta fallback, Esc to quit");
            self.hint_logged = true;
        }
        let now = match self.albedo.peek() {
            HandleState::Loading => LoggedState::Loading,
            HandleState::Ready(_) => LoggedState::Ready,
            HandleState::Failed(e) => {
                if self.last_logged_state != LoggedState::Failed {
                    eprintln!("assets demo: albedo load failed: {e}");
                }
                LoggedState::Failed
            }
        };
        if now != self.last_logged_state {
            match now {
                LoggedState::Loading => {} // start state; not surprising
                LoggedState::Ready => {
                    println!("assets demo: albedo loaded — cube rebinds to real texture")
                }
                LoggedState::Failed => {} // already printed above
            }
            self.last_logged_state = now;
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
        self.camera_binding.write(&renderer.queue, &self.camera);

        self.buckets.begin_frame();
        for (zone_id, zone) in sim.zones.iter() {
            for (id, obj) in zone.iter() {
                let model = Mat4::from_rotation_translation(obj.rotation, obj.position);
                let hit_id = renderer.reserve_object(zone_id, id);
                self.buckets.push(
                    MatId::Cube,
                    MeshInstanceAttribs::new(model, Vec4::ONE).with_hit_id(hit_id),
                );
            }
        }
        self.buckets.upload(&renderer.queue);

        // Reconcile the material instance with its handle's current state.
        // Cheap when nothing changed; fires the bind-group rebuild on the
        // frame the load transitions to `Ready` (or when the user
        // press/releases F to flip the force-loading toggle).
        if let Some(instance) = self.instances.get_mut(MatId::Cube) {
            instance.refresh(renderer, &self.material, &self.samplers, &self.asset_server);
        }

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
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        match (code, event.state) {
            (KeyCode::Escape, ElementState::Pressed) => ctx.event_loop.exit(),
            (KeyCode::KeyF, ElementState::Pressed) => self.asset_server.set_force_loading(true),
            (KeyCode::KeyF, ElementState::Released) => self.asset_server.set_force_loading(false),
            _ => {}
        }
    }
}

fn main() {
    currawong::run::<AssetsView>(Game::new());
}
