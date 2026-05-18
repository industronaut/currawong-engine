//! Blender-authored multi-primitive glb + name-resolved materials (#80).
//!
//! Streams [`assets/lumber/base.glb`](../assets/lumber/base.glb) through the
//! asset server, decoding it into a `Vec<MeshPrimitive>` (one per glb
//! primitive, i.e. one per Blender material slot) with each primitive
//! carrying its material slot name as a string. At draw time the example
//! looks each primitive's name up against a [`MaterialRegistry`] and binds
//! whichever instance is registered; misses fall back to a magenta-tinted
//! `PbrMaterialInstance` the example wires up at init.
//!
//! ## What the asset looks like
//!
//! `base.glb` was authored in Blender, exported with the default glTF
//! settings. Its single mesh has one primitive (one material slot, named
//! `"Lumber"` on the glb side). On a fresh checkout that slot name does
//! *not* satisfy [`MaterialId`]'s `namespace:name` grammar, so the
//! registry miss path runs and the model renders in magenta —
//! deliberately, to make the missing-binding case loud. To wire a real
//! material:
//!
//! 1. In Blender, rename the material slot from `Lumber` to
//!    `currawong:mat_lumber` (or any `namespace:name` that the registry
//!    knows). Re-export the glb.
//! 2. The registry already has a `currawong:mat_lumber` entry tuned to
//!    warm wood (registered below). The next frame the model rebinds to
//!    the wood material.
//!
//! ## What this exercises
//!
//! - Multi-primitive decode (`decode_gltf_mesh` returns
//!   `Vec<DecodedPrimitive>`; the loader no longer rejects multi-slot
//!   exports).
//! - Y-up → Z-up axis flip at decode time (Blender writes Y-up per the
//!   glTF spec; we want engine-native Z-up). The model's height ends up
//!   along +Z without per-frame fixups.
//! - Name-based material binding through [`MaterialRegistry`], with the
//!   magenta fallback exercising the missing-binding path.
//! - The PR-2 `set_force_loading` debug toggle still works — holding `F`
//!   substitutes the asset-server fallbacks (magenta texture, unit cube
//!   sized to the template's visual AABB) for every handle.
//!
//! ## Controls
//!
//! - `F` (hold) — force every handle to look `Loading` (PR-2 toggle).
//! - `R` (hold) — force the *material registry* to ignore every glb
//!   primitive's name and resolve to the magenta fallback instance,
//!   regardless of whether a real material was registered. Useful for
//!   inspecting how a hypothetical missing-name primitive would render.
//! - Right-click drag — orbit camera.
//! - WASD — pan camera focus.
//! - Scroll — zoom.
//! - Esc — quit.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use currawong::data::{FsSource, Vfs, VfsPath};
use currawong::glam::{Mat4, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Camera, CameraBinding, EngineCtx, Handle, MaterialId, MaterialRegistry,
    Mesh, MeshInstanceAttribs, OrbitRig, PbrMaterial, PbrMaterialInstance, PbrMaterialParams,
    Renderer, SamplerKind, SamplerRegistry, Simulation, Texture, View, ViewConfig, ViewEnvironment,
    Zone, ZoneId, Zones, sun_direction_for, wgpu, winit,
};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// --- Sim ---------------------------------------------------------------

/// Empty sim with one zone + a time-of-day so the sun rotates while we
/// stare at the building. The whole example is view-centric — the import
/// pipeline is a view concern.
struct Game {
    #[allow(dead_code)]
    zones: Zones,
    zone: ZoneId,
    time_of_day: f32,
}

impl Game {
    fn new() -> Self {
        let mut zones = Zones::new();
        let zone = zones.insert(Zone::new());
        Self {
            zones,
            zone,
            time_of_day: 0.35,
        }
    }
}

impl Simulation for Game {
    fn tick(&mut self, dt: Duration) {
        let day_seconds = 180.0;
        self.time_of_day = (self.time_of_day + dt.as_secs_f32() / day_seconds).rem_euclid(1.0);
    }
}

// --- View --------------------------------------------------------------

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct BlenderImportView {
    camera: Camera,
    camera_binding: CameraBinding,
    rig: OrbitRig,

    material: PbrMaterial,
    samplers: SamplerRegistry,
    asset_server: AssetServer,

    /// Name-keyed material registry. The example registers one real
    /// material under `currawong:mat_lumber` (a warm wood tone). Any glb
    /// primitive whose slot name resolves here binds the real instance;
    /// everything else routes through [`fallback_material`](Self::fallback_material).
    materials: MaterialRegistry<PbrMaterialInstance>,

    /// Magenta-flavoured PBR instance bound when [`materials`] misses on
    /// a primitive's slot name. Lives outside the registry so misses can
    /// resolve to it unconditionally without polluting the registry's
    /// `iter()`.
    fallback_material: PbrMaterialInstance,

    mesh_handle: Handle<Mesh>,
    /// Visual bounds used to size the unit-cube fallback while the real
    /// mesh is still streaming. Same shape as a real
    /// [`RenderTemplate`](currawong::RenderTemplate)'s declared AABB.
    visual_bounds: Aabb,
    instance_buffer: wgpu::Buffer,

    force_registry_miss: bool,
    hint_logged: bool,
    last_resolved_names: Vec<Option<String>>,
}

impl View for BlenderImportView {
    type Sim = Game;

    const CONFIG: ViewConfig = ViewConfig {
        title: "currawong — blender_import (multi-primitive glb + name-bound materials)",
        clear_colour: wgpu::Color {
            r: 0.10,
            g: 0.12,
            b: 0.16,
            a: 1.0,
        },
        depth_format: Some(DEPTH_FORMAT),
    };

    fn init(renderer: &Renderer) -> Self {
        let camera = Camera {
            position: Vec3::new(3.5, -4.5, 3.5),
            target: Vec3::new(0.0, 0.0, 0.5),
            ..Camera::default()
        };
        let camera_binding = CameraBinding::new(&renderer.device);
        let mut rig = OrbitRig::new(Vec3::new(0.0, 0.0, 0.5));
        rig.distance = 5.5;
        rig.pitch = 35.0_f32.to_radians();

        let samplers = SamplerRegistry::new(&renderer.device);
        let material = PbrMaterial::new(renderer, camera_binding.layout());

        // Asset VFS: just the repo's `assets/` directory, same mount the
        // other examples use.
        let assets_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        let mut vfs = Vfs::new();
        vfs.mount(FsSource::new(assets_root));
        let asset_server = AssetServer::new(renderer, Arc::new(vfs));

        // Stream the Blender export. `decode_gltf_mesh` will surface its
        // primitives and each primitive's material slot name; we don't
        // need to do anything special at handle-request time.
        let mesh_handle = asset_server.mesh(VfsPath::new("lumber/base.glb").expect("valid path"));

        // Visual bounds for the fallback sizing while the glb is still
        // loading. base.glb is a small building roughly 2×2 m footprint,
        // ~1 m tall in engine-native Z-up.
        let visual_bounds = Aabb::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

        // --- Material registry --------------------------------------------
        //
        // One real material under the conventional name, plus a magenta
        // fallback kept outside the registry. The Blender export's slot
        // name on a fresh checkout is `"Lumber"` (no namespace), which
        // misses the registry by construction — the fallback path runs
        // until the artist re-exports as `currawong:mat_lumber`.
        let mut materials: MaterialRegistry<PbrMaterialInstance> = MaterialRegistry::new();
        let lumber_albedo = Texture::from_rgba8(
            renderer,
            "lumber albedo 1x1",
            1,
            1,
            // Warm pine-board tone, deliberately flat so re-exports with
            // a real texture map override it cleanly.
            &[180, 140, 90, 255],
            true,
        );
        let lumber = material.create_instance(
            renderer,
            &samplers,
            &asset_server,
            PbrMaterialParams {
                albedo: Handle::ready(lumber_albedo),
                sampler: SamplerKind::LinearRepeat,
                albedo_factor: Vec4::ONE,
                metallic: 0.0,
                roughness: 0.65,
            },
        );
        materials.register(
            MaterialId::new("currawong:mat_lumber").expect("valid id"),
            lumber,
        );

        let magenta_albedo = Texture::from_rgba8(
            renderer,
            "fallback magenta 1x1",
            1,
            1,
            &[255, 0, 255, 255],
            true,
        );
        let fallback_material = material.create_instance(
            renderer,
            &samplers,
            &asset_server,
            PbrMaterialParams {
                albedo: Handle::ready(magenta_albedo),
                sampler: SamplerKind::NearestClamp,
                albedo_factor: Vec4::ONE,
                metallic: 0.0,
                // Higher roughness so the magenta reads as flat — easier
                // to recognise as "missing material" than a glossy one.
                roughness: 0.9,
            },
        );

        let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blender_import instance"),
            size: std::mem::size_of::<MeshInstanceAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            camera,
            camera_binding,
            rig,
            material,
            samplers,
            asset_server,
            materials,
            fallback_material,
            mesh_handle,
            visual_bounds,
            instance_buffer,
            force_registry_miss: false,
            hint_logged: false,
            last_resolved_names: Vec::new(),
        }
    }

    fn active_zone(&self, sim: &Game) -> Option<ZoneId> {
        Some(sim.zone)
    }

    fn extract_environment(&self, sim: &Game, _zone: ZoneId) -> ViewEnvironment {
        ViewEnvironment {
            sun_direction: sun_direction_for(sim.time_of_day),
            sun_color: Vec3::splat(2.2),
            ambient: Vec3::new(0.30, 0.32, 0.38),
            sky_color: Vec3::new(0.45, 0.55, 0.75),
        }
    }

    fn update(&mut self, _sim: &Game, _ctx: &mut EngineCtx, dt: Duration) {
        self.rig.update(dt);
        self.rig.apply_to(&mut self.camera);

        if !self.hint_logged {
            println!(
                "blender_import: hold F to force the magenta texture fallback, \
                 R to force registry misses, Esc to quit"
            );
            println!(
                "blender_import: registered materials = {}; \
                 fallback path runs when a glb primitive's slot name doesn't \
                 parse as `namespace:name` or isn't registered",
                self.materials.len()
            );
            self.hint_logged = true;
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

        // Reconcile both material bind groups with the texture-handle
        // states. Cheap in steady state.
        for (_id, instance) in self.materials.iter_mut() {
            instance.refresh(renderer, &self.material, &self.samplers, &self.asset_server);
        }
        self.fallback_material.refresh(
            renderer,
            &self.material,
            &self.samplers,
            &self.asset_server,
        );

        // Resolve the mesh handle. While loading the fallback is a single-
        // primitive unit cube (material_name = None → registry miss →
        // magenta fallback).
        let resolved = self
            .asset_server
            .resolve_mesh(&self.mesh_handle, Some(self.visual_bounds));
        let model = Mat4::from_translation(Vec3::ZERO) * resolved.fallback_adjustment;
        let attribs = MeshInstanceAttribs::new(model, Vec4::ONE);
        renderer
            .queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::bytes_of(&attribs));

        pass.set_pipeline(self.material.pipeline());
        pass.set_bind_group(0, self.camera_binding.bind_group(), &[]);
        pass.set_bind_group(1, renderer.scene_bind_group(), &[]);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        // Per-primitive draw: bind the material the registry resolves
        // to, then draw. The registry takes the glb material slot name
        // verbatim; misses fall back to the magenta instance.
        let mut frame_names: Vec<Option<String>> = Vec::with_capacity(resolved.primitives.len());
        for prim in resolved.primitives {
            let raw = prim.material_name.as_deref();
            let resolved_real = (!self.force_registry_miss)
                .then(|| raw.and_then(|n| self.materials.get_by_name(n)))
                .flatten();
            let instance = resolved_real.unwrap_or(&self.fallback_material);
            pass.set_bind_group(2, instance.bind_group(), &[]);
            pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
            pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..prim.index_count, 0, 0..1);
            frame_names.push(prim.material_name.clone());
        }

        // Log transitions in primitive material names so the dev sees
        // exactly which slots the glb declared as the load lands.
        if frame_names != self.last_resolved_names {
            println!(
                "blender_import: glb primitives now declare material slots {:?}",
                frame_names,
            );
            self.last_resolved_names = frame_names;
        }
    }

    fn input(&mut self, _: &mut Game, ctx: &mut EngineCtx, event: &WindowEvent) {
        self.rig.handle_event(event);
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
            (KeyCode::KeyR, ElementState::Pressed) => self.force_registry_miss = true,
            (KeyCode::KeyR, ElementState::Released) => self.force_registry_miss = false,
            _ => {}
        }
    }
}

fn main() {
    currawong::run::<BlenderImportView>(Game::new());
}
