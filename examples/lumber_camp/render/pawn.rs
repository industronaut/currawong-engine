//! View-side state and helpers specific to drawing pawns.
//!
//! Owns the pawn-only resources — tick-boundary interpolation snapshots,
//! the wall-clock idle-bob phase, and the carried-log mesh/buffer — plus
//! the per-frame helpers the top-level fused render walk dispatches into.
//! The pawn *body* template (the satchel'd capsule) is built here too but
//! still registered in the central `templates` map by `LumberCampView::init`
//! so the main bucket draw loop stays uniform across kinds.
//!
//! The fused render walk in [`super`] preserves the single-walk shape (cache
//! and i-cache stay friendly) by calling [`PawnRenderer::position_for`] and
//! [`PawnRenderer::push_log_if_carrying`] inline per pawn rather than
//! handing this module the zone iterator.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    EngineCtx, MeshInstanceAttribs, PbrMaterial, PosNormalUv, PrimitiveMesh, Renderer,
    SamplerRegistry, Texture, WorldObjectId, Zone, wgpu,
};

use super::{MeshTemplate, TemplateParams};
use crate::sim::{Carrying, Game, Move, RenderId};

/// Vertical amplitude of the idle bob applied to pawns without a Move.
/// A few centimetres reads as breathing without looking like a glitch.
const IDLE_BOB_AMPLITUDE: f32 = 0.035;
/// Idle-bob frequency in Hz. Slow enough to feel like a breath rather
/// than a hop; wall-clock driven so paused pawns still breathe.
const IDLE_BOB_HZ: f32 = 1.4;
/// Upper bound on simultaneously-carried logs (= number of pawns in flight
/// to the stockpile). Sized for the log instance buffer.
const MAX_LOGS: u32 = 32;

/// Build the pawn *body* template (mesh + PBR material instance). Called
/// from [`super::LumberCampView::init`] and registered in the central
/// templates map so the main bucket draw loop renders pawns alongside other
/// kinds without special-casing.
pub fn new_body_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    albedo: &Texture,
) -> MeshTemplate {
    MeshTemplate::new(
        renderer,
        material,
        samplers,
        albedo,
        &pawn_mesh_with_satchel(),
        TemplateParams {
            label: "lumber-camp pawn",
            albedo_factor: Vec4::new(0.95, 0.70, 0.55, 1.0), // skin-warm
            metallic: 0.0,
            roughness: 0.70,
        },
    )
}

/// Per-frame pawn-only state owned by the view. Holds the carried-log
/// template + its instance buffer, the wall-clock origin for the idle bob,
/// and the tick-boundary position snapshots used to interpolate pawn motion
/// at any render rate.
pub struct PawnRenderer {
    log_template: MeshTemplate,
    log_buffer: wgpu::Buffer,
    log_scratch: Vec<MeshInstanceAttribs>,

    /// Pawn positions captured at the previous tick boundary. With
    /// [`pawn_curr`](Self::pawn_curr) and the current tick's `alpha`,
    /// [`position_for`](Self::position_for) lerps pawn world-positions for
    /// smooth motion at any render rate. Non-pawn objects don't move; they
    /// draw straight from the live sim transform.
    pawn_prev: HashMap<WorldObjectId, Vec3>,
    pawn_curr: HashMap<WorldObjectId, Vec3>,
    /// `SimClock::total_ticks` at the last `update` that took a snapshot;
    /// snapshots only roll over when this changes, so paused frames keep
    /// `prev` and `curr` matched and the lerp pins to the sim position.
    last_seen_tick: u64,

    /// Wall-clock origin for the idle-bob phase. Wall-clock — not sim time —
    /// so the bob keeps breathing while the sim is paused.
    started: Instant,
    /// Cached bob offset for this frame, recomputed once in `begin_frame`
    /// so it doesn't drift between the two pawns drawn in the same frame.
    frame_bob_offset: f32,

    /// Local-frame transform for the carried log: lay the cylinder on its
    /// side (rotate around X) and sit it at shoulder height. The log model
    /// is `pawn_model * log_local`, so it inherits the pawn's interpolated
    /// position and facing for free.
    log_local: Mat4,
}

impl PawnRenderer {
    pub fn new(
        renderer: &Renderer,
        material: &PbrMaterial,
        samplers: &SamplerRegistry,
        albedo: &Texture,
    ) -> Self {
        let log_mesh = PrimitiveMesh::cylinder(0.07, 0.6, 12, true);
        let log_template = MeshTemplate::new(
            renderer,
            material,
            samplers,
            albedo,
            &log_mesh,
            TemplateParams {
                label: "lumber-camp carried log",
                albedo_factor: Vec4::new(0.42, 0.27, 0.16, 1.0), // wood brown
                metallic: 0.0,
                roughness: 0.85,
            },
        );
        let log_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lumber-camp log instances"),
            size: u64::from(MAX_LOGS) * std::mem::size_of::<MeshInstanceAttribs>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let log_local = Mat4::from_rotation_translation(
            Quat::from_rotation_x(PI / 2.0),
            Vec3::new(0.0, 0.0, 0.95),
        );
        Self {
            log_template,
            log_buffer,
            log_scratch: Vec::with_capacity(MAX_LOGS as usize),
            pawn_prev: HashMap::new(),
            pawn_curr: HashMap::new(),
            last_seen_tick: 0,
            started: Instant::now(),
            frame_bob_offset: 0.0,
            log_local,
        }
    }

    /// Per-frame snapshot rollover. The standard fixed-tick interpolation
    /// pattern: previous tick's positions are the lerp source, this tick's
    /// positions are the destination, and `alpha` picks where between them
    /// the render call draws. Snapshot only when the tick counter has
    /// actually advanced, so paused / between-tick frames don't churn the
    /// maps.
    pub fn update(&mut self, sim: &Game, ctx: &EngineCtx) {
        let now_tick = ctx.clock.total_ticks();
        if now_tick == self.last_seen_tick {
            return;
        }
        self.pawn_prev = std::mem::take(&mut self.pawn_curr);
        if let Some(zone) = sim.zones.get(sim.zone) {
            for (id, transform) in zone.iter() {
                if zone.components().get::<RenderId>(id) == Some(&RenderId::Pawn) {
                    self.pawn_curr.insert(id, transform.position);
                }
            }
        }
        self.last_seen_tick = now_tick;
    }

    /// Reset per-frame scratch and recompute the idle-bob offset for this
    /// frame. Called once by the top-level view at the start of each render.
    pub fn begin_frame(&mut self) {
        self.log_scratch.clear();
        let phase = self.started.elapsed().as_secs_f32() * IDLE_BOB_HZ * TAU;
        self.frame_bob_offset = phase.sin() * IDLE_BOB_AMPLITUDE;
    }

    /// World-space position for a pawn this frame: lerp between the two most
    /// recent tick-boundary snapshots, then add the idle-bob offset if the
    /// pawn currently has no [`Move`].
    pub fn position_for(
        &self,
        zone: &Zone,
        id: WorldObjectId,
        live_position: Vec3,
        alpha: f32,
    ) -> Vec3 {
        let prev = self.pawn_prev.get(&id).copied().unwrap_or(live_position);
        let curr = self.pawn_curr.get(&id).copied().unwrap_or(live_position);
        let mut pos = prev.lerp(curr, alpha);
        if zone.components().get::<Move>(id).is_none() {
            pos.z += self.frame_bob_offset;
        }
        pos
    }

    /// If the pawn at `id` has [`Carrying`], queue a log instance whose
    /// model matrix is `pawn_model * log_local`. Inheriting from
    /// `pawn_model` means the log picks up interpolation and facing without
    /// re-deriving them here.
    pub fn push_log_if_carrying(&mut self, zone: &Zone, id: WorldObjectId, pawn_model: Mat4) {
        if zone.components().get::<Carrying>(id).is_none() {
            return;
        }
        if self.log_scratch.len() >= MAX_LOGS as usize {
            return;
        }
        let log_model = pawn_model * self.log_local;
        self.log_scratch
            .push(MeshInstanceAttribs::new(log_model, Vec4::ONE));
    }

    /// Upload this frame's log instances. No-op when nothing was queued.
    pub fn upload_logs(&self, queue: &wgpu::Queue) {
        if self.log_scratch.is_empty() {
            return;
        }
        queue.write_buffer(&self.log_buffer, 0, bytemuck::cast_slice(&self.log_scratch));
    }

    /// Draw the queued logs. Caller must have bound the pawn/PBR pipeline
    /// and the camera+scene bind groups already; this binds the log
    /// material/mesh/instance buffers and issues the indexed-instanced draw.
    pub fn draw_logs(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.log_scratch.is_empty() {
            return;
        }
        pass.set_bind_group(2, self.log_template.material.bind_group(), &[]);
        pass.set_vertex_buffer(0, self.log_template.vertices.slice(..));
        pass.set_vertex_buffer(1, self.log_buffer.slice(..));
        pass.set_index_buffer(
            self.log_template.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(
            0..self.log_template.index_count,
            0,
            0..self.log_scratch.len() as u32,
        );
    }
}

/// Pawn body + a small offset cube ("satchel") on the local +X side, baked
/// into one mesh. Same material as the body, so the satchel doesn't visually
/// pop on its own — but the geometric protrusion catches the sun at a
/// different angle than the capsule surface, making the pawn's facing
/// readable as it walks around.
fn pawn_mesh_with_satchel() -> PrimitiveMesh {
    let mut mesh = PrimitiveMesh::capsule(0.30, 1.6, 16, 3);
    let satchel = PrimitiveMesh::cube(Vec3::splat(0.18));
    let offset = Vec3::new(0.34, 0.0, 0.45);
    let base = mesh.vertices.len() as u32;
    mesh.vertices
        .extend(satchel.vertices.iter().map(|v| PosNormalUv {
            position: [
                v.position[0] + offset.x,
                v.position[1] + offset.y,
                v.position[2] + offset.z,
            ],
            normal: v.normal,
            uv: v.uv,
        }));
    mesh.indices
        .extend(satchel.indices.iter().map(|&i| base + i));
    mesh
}
