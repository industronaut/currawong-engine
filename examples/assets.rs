//! Streaming asset loading, demonstrated.
//!
//! Two PBR cubes side by side, both wired through the [`AssetServer`].
//! The left cube's *mesh* is built in code (the existing
//! [`PrimitiveMesh::cube`] generator) — only its *albedo texture* streams
//! through the VFS. The right cube's mesh streams too, from a tiny
//! checked-in `assets/test/cube.glb`; both cubes share the same texture
//! handle, so on a cold start you watch the right cube transition from
//! the asset server's magenta unit-cube fallback (sized to the template's
//! declared visual AABB) to the real glTF geometry.
//!
//! ## Controls
//!
//! - `F` (hold) — force every handle to *look* `Loading` to consumers.
//!   Both cubes rebind to their respective fallbacks (texture for the
//!   left, mesh for the right; the right also loses the texture).
//! - `Esc` — quit.
//!
//! ## What this proves
//!
//! - `Handle::peek` returns `Loading` until the background thread
//!   publishes, then `Ready` for the rest of the run — separately for
//!   each asset type, so the texture can land before the mesh or
//!   vice versa.
//! - [`AssetServer::resolve_mesh`] returns the unit-cube fallback buffers
//!   plus a translate+scale matrix sized to the template's visual AABB.
//!   Compose it into the per-instance model and the placeholder
//!   occupies the same volume the real mesh will.
//! - The PR-2 debug toggle now exercises *both* fallback paths in one
//!   keybind: holding `F` flips the asset server into forced-fallback
//!   mode, which `resolve_texture` *and* `resolve_mesh` both honour.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{FsSource, Vfs, VfsPath};
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Camera, CameraBinding, CommandQueue, EngineCtx, Handle, HandleState, Mesh,
    MeshInstanceAttribs, PbrMaterial, PbrMaterialInstance, PbrMaterialParams, PrimitiveMesh,
    Renderer, SamplerKind, SamplerRegistry, SimUnit, Simulation, Texture, TextureColorSpace, View,
    ViewConfig, ViewEnvironment, Zone, ZoneId, Zones, sun_direction_for, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ----------------------------------------------------------------

/// Minimal sim: a `time_of_day` advancing the sun direction, plus one
/// empty zone so `View::active_zone` has something to point at. This
/// example is deliberately view-centric — the asset pipeline is a
/// view-side concern; sim-driven object iteration is the lumber camp
/// example's job.
struct Game {
    #[allow(dead_code)]
    zones: Zones,
    zone: ZoneId,
    time_of_day: SimUnit,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        Self {
            zones,
            zone: zone_id,
            time_of_day: SimUnit::from_num(0.30),
        }
    }
}

impl Simulation for Game {
    type Command = ();
    fn tick(&mut self, dt: Duration) {
        // dt → Q16.16 sim-seconds via integer nanos; bit-exact.
        let nanos = dt.as_nanos();
        let secs_q16: u128 = (nanos << 16) / 1_000_000_000;
        let dt_secs = SimUnit::from_bits(secs_q16.min(i32::MAX as u128) as i32);
        let day_seconds = SimUnit::from_num(90);
        let next = self.time_of_day + dt_secs / day_seconds;
        // rem_euclid against 1.0 via manual wrap.
        let mut wrapped = next;
        while wrapped >= SimUnit::from_num(1) {
            wrapped -= SimUnit::from_num(1);
        }
        while wrapped < SimUnit::ZERO {
            wrapped += SimUnit::from_num(1);
        }
        self.time_of_day = wrapped;
    }
}

// --- View ---------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct AssetsView {
    camera: Camera,
    camera_binding: CameraBinding,
    material: PbrMaterial,
    samplers: SamplerRegistry,
    asset_server: AssetServer,
    /// Both cubes share one material instance — same texture handle, same
    /// sampler, same factors. Only the geometry binding differs per draw.
    cube_material: PbrMaterialInstance,
    inline_cube: InlineCube,
    gltf_cube: GltfCube,
    last_texture_state: LoggedState,
    last_mesh_state: LoggedState,
    hint_logged: bool,
}

/// Left-hand cube — vertex/index buffers built once at init from the
/// in-code [`PrimitiveMesh::cube`] generator. Drawn unconditionally.
struct InlineCube {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// Per-instance attribute buffer rewritten each frame so the cube can
    /// drift / rotate / change tint without rebuilding pipelines.
    instance_buffer: wgpu::Buffer,
    position: Vec3,
}

/// Right-hand cube — mesh streamed through the asset server. The handle
/// starts in `Loading` and the per-frame [`AssetServer::resolve_mesh`]
/// call substitutes the unit-cube fallback (scaled to `visual_bounds`)
/// until the real mesh lands.
struct GltfCube {
    handle: Handle<Mesh>,
    instance_buffer: wgpu::Buffer,
    position: Vec3,
    /// What [`AssetServer::resolve_mesh`] uses to size the fallback
    /// placeholder. Same shape as a [`RenderTemplate`](currawong::RenderTemplate)'s
    /// declared visual AABB — kept here so this example doesn't need the
    /// full templating machinery just to demonstrate the resolve API.
    visual_bounds: Aabb,
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
        ..ViewConfig::DEFAULT
    };

    fn init(renderer: &Renderer) -> Self {
        use wgpu::util::DeviceExt;

        let device = &renderer.device;
        let camera = Camera {
            position: Vec3::new(3.5, -4.5, 3.5),
            target: Vec3::new(0.0, 0.0, 0.5),
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(device);
        let samplers = SamplerRegistry::new(device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());

        let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mut vfs = Vfs::new();
        vfs.mount(FsSource::new(assets_root));
        let asset_server = AssetServer::new(renderer, Arc::new(vfs));

        // Streamed texture — bound by the shared material instance. Both
        // cubes will sample it once the load lands.
        let albedo: Handle<Texture> = asset_server.texture(
            VfsPath::new("checker_albedo.png").expect("valid VFS path"),
            TextureColorSpace::Srgb,
        );

        let cube_material = material.create_instance(
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

        // --- Inline cube --------------------------------------------------
        let inline_mesh = PrimitiveMesh::cube(Vec3::ONE);
        let inline_vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inline cube vertices"),
            contents: bytemuck::cast_slice(&inline_mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let inline_index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inline cube indices"),
            contents: bytemuck::cast_slice(&inline_mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let inline_cube = InlineCube {
            vertex_buffer: inline_vertex,
            index_buffer: inline_index,
            index_count: inline_mesh.index_count(),
            instance_buffer: create_one_instance_buffer(device, "inline cube instance"),
            position: Vec3::new(-1.0, 0.0, 0.5),
        };

        // --- glTF cube ----------------------------------------------------
        let gltf_handle = asset_server.mesh(VfsPath::new("test/cube.glb").expect("valid VFS path"));
        let gltf_cube = GltfCube {
            handle: gltf_handle,
            instance_buffer: create_one_instance_buffer(device, "gltf cube instance"),
            position: Vec3::new(1.0, 0.0, 0.5),
            // Declare the visual AABB *we know* the glTF cube has so the
            // fallback placeholder is sized to match. The asset server
            // can't infer this from a Loading handle — that's the whole
            // reason `resolve_mesh` takes bounds as a parameter.
            visual_bounds: Aabb::cube(0.5),
        };

        Self {
            camera,
            camera_binding,
            material,
            samplers,
            asset_server,
            cube_material,
            inline_cube,
            gltf_cube,
            last_texture_state: LoggedState::Loading,
            last_mesh_state: LoggedState::Loading,
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
            ..ViewEnvironment::neutral()
        }
    }

    fn update(
        &mut self,
        _sim: &Game,
        _ctx: &mut EngineCtx,
        _: &mut CommandQueue<()>,
        _dt: Duration,
    ) {
        if !self.hint_logged {
            println!("assets demo: hold F to force the magenta fallback, Esc to quit");
            self.hint_logged = true;
        }

        // Log texture-handle state transitions (PR-2 behaviour).
        let texture_state = match self.cube_material.handle().peek() {
            HandleState::Loading => LoggedState::Loading,
            HandleState::Ready(_) => LoggedState::Ready,
            HandleState::Failed(e) => {
                if self.last_texture_state != LoggedState::Failed {
                    eprintln!("assets demo: albedo load failed: {e}");
                }
                LoggedState::Failed
            }
        };
        if texture_state != self.last_texture_state {
            if texture_state == LoggedState::Ready {
                println!("assets demo: albedo loaded — cubes rebind to real texture");
            }
            self.last_texture_state = texture_state;
        }

        // Log mesh-handle state transitions (PR-3 addition).
        let mesh_state = match self.gltf_cube.handle.peek() {
            HandleState::Loading => LoggedState::Loading,
            HandleState::Ready(_) => LoggedState::Ready,
            HandleState::Failed(e) => {
                if self.last_mesh_state != LoggedState::Failed {
                    eprintln!("assets demo: glTF mesh load failed: {e}");
                }
                LoggedState::Failed
            }
        };
        if mesh_state != self.last_mesh_state {
            if mesh_state == LoggedState::Ready {
                println!(
                    "assets demo: glTF mesh loaded — right cube swaps from placeholder to real geometry"
                );
            }
            self.last_mesh_state = mesh_state;
        }
    }

    fn render(
        &mut self,
        _sim: &Game,
        _alpha: f32,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let size = renderer.window.inner_size();
        if size.height > 0 {
            self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        }
        self.camera_binding.write(&renderer.queue, &self.camera);

        // Reconcile the shared material instance with its handle's current
        // state. Cheap when nothing changed; fires the bind-group rebuild
        // on the frame the texture transitions to `Ready` (or when the
        // user press/releases F to flip the force-loading toggle).
        self.cube_material
            .refresh(renderer, &self.material, &self.samplers, &self.asset_server);

        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        pass.set_bind_group(2, self.cube_material.bind_group(), &[]);

        // Inline cube: a single instance at its position. Picking IDs left
        // at the 0 default — neither cube is clickable in this demo.
        let inline_model = Mat4::from_translation(self.inline_cube.position);
        let inline_attribs = MeshInstanceAttribs::new(inline_model, Vec4::ONE);
        renderer.queue.write_buffer(
            &self.inline_cube.instance_buffer,
            0,
            bytemuck::bytes_of(&inline_attribs),
        );
        pass.set_vertex_buffer(0, self.inline_cube.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.inline_cube.instance_buffer.slice(..));
        pass.set_index_buffer(
            self.inline_cube.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..self.inline_cube.index_count, 0, 0..1);

        // glTF cube: resolve the handle, compose the fallback adjustment
        // inside the per-instance model, draw each primitive in the
        // resolved mesh. The cube.glb test fixture has a single primitive
        // — multi-primitive support (#80) loops the same code without
        // change.
        let resolved = self
            .asset_server
            .resolve_mesh(&self.gltf_cube.handle, Some(self.gltf_cube.visual_bounds));
        let gltf_model =
            Mat4::from_translation(self.gltf_cube.position) * resolved.fallback_adjustment;
        let gltf_attribs = MeshInstanceAttribs::new(gltf_model, Vec4::ONE);
        renderer.queue.write_buffer(
            &self.gltf_cube.instance_buffer,
            0,
            bytemuck::bytes_of(&gltf_attribs),
        );
        pass.set_vertex_buffer(1, self.gltf_cube.instance_buffer.slice(..));
        for prim in resolved.primitives {
            pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
            pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..prim.index_count, 0, 0..1);
        }
    }

    fn input(
        &mut self,
        _: &Game,
        ctx: &mut EngineCtx,
        _: &mut CommandQueue<()>,
        event: &WindowEvent,
    ) {
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

/// Allocate a `wgpu::Buffer` sized for one [`MeshInstanceAttribs`], with
/// `COPY_DST` so we can rewrite it each frame. The shared shape across
/// both cubes — one instance buffer per cube means consecutive
/// `queue.write_buffer` calls don't clobber each other within a frame.
fn create_one_instance_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<MeshInstanceAttribs>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn main() {
    currawong::run::<AssetsView>(Game::new());
}
