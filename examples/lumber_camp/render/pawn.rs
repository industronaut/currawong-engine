//! View-side state and helpers specific to drawing pawns.
//!
//! The pawn body and the carried log are both [`MeshPart`](currawong::MeshPart)s
//! on the engine [`RenderTemplate`](currawong::RenderTemplate) registered
//! by [`super::LumberCampView::init`]. This module exports the factory
//! functions for each part's mesh + PBR material instance, the log's
//! local-frame transform, and the pawn's visual AABB.
//!
//! View-side pawn state lives on [`PawnRenderer`]: tick-boundary position
//! snapshots for sub-tick interpolation, and the wall-clock idle-bob
//! phase. Both are read by the per-instance update closure in [`super`],
//! which overwrites `instance.world_xform` with the interpolated +
//! bobbed pose; the engine then composes `world_xform *
//! log_local_transform` for the log part automatically.
//!
//! Log visibility is also a view-side decision, set in that same closure:
//! `instance.mesh_parts[LOG_PART].visible = components.get::<Carrying>(id).is_some()`.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, EngineCtx, PbrMaterial, PosNormalUv, PrimitiveMesh, Renderer, SamplerRegistry, Texture,
    WorldObjectId,
};

use super::{MeshTemplate, TemplateParams};
use crate::sim::{Game, RenderId};

/// Index of the carried-log mesh part in the pawn render template. Stable
/// because parts keep their declaration order on [`RenderTemplate`].
pub const LOG_PART: usize = 1;

/// Vertical amplitude of the idle bob applied to pawns without a [`Move`](crate::sim::Move).
/// A few centimetres reads as breathing without looking like a glitch.
const IDLE_BOB_AMPLITUDE: f32 = 0.035;
/// Idle-bob frequency in Hz. Slow enough to feel like a breath rather
/// than a hop; wall-clock driven so paused pawns still breathe.
const IDLE_BOB_HZ: f32 = 1.4;

/// Build the pawn *body* template (capsule + satchel, skin-warm PBR).
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

/// Build the carried-log template — a wood-brown horizontal cylinder.
pub fn new_log_template(
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
        &PrimitiveMesh::cylinder(0.07, 0.6, 12, true),
        TemplateParams {
            label: "lumber-camp carried log",
            albedo_factor: Vec4::new(0.42, 0.27, 0.16, 1.0), // wood brown
            metallic: 0.0,
            roughness: 0.85,
        },
    )
}

/// Local-frame transform for the carried log: lay the cylinder on its
/// side and sit it at shoulder height in the pawn's local frame. Engine
/// composes `parent.world_xform * log_local_transform()`, so the log
/// rides whatever pose the per-instance update writes onto the proxy.
pub fn log_local_transform() -> Mat4 {
    Mat4::from_rotation_translation(Quat::from_rotation_x(PI / 2.0), Vec3::new(0.0, 0.0, 0.95))
}

/// Visual AABB in the pawn's local frame. Encloses the 1.6 m capsule
/// centred on the origin plus the satchel protrusion plus the carried log
/// at shoulder height — the proxy frustum-culls against this even when
/// the pawn isn't carrying anything (cheaper than swapping bounds with
/// the `Carrying` flag, and grazing-edge correct).
pub fn visual_bounds() -> Aabb {
    Aabb::new(Vec3::new(-0.40, -0.40, -0.85), Vec3::new(0.70, 0.40, 1.10))
}

/// Per-frame pawn-only view state: tick-boundary position snapshots for
/// sub-tick interpolation, and the wall-clock idle-bob phase shared by
/// every pawn drawn this frame.
pub struct PawnRenderer {
    /// Pawn positions at the previous tick boundary. With
    /// [`pawn_curr`](Self::pawn_curr) and the current tick's `alpha`,
    /// [`interp_position`](Self::interp_position) lerps for smooth motion
    /// at any render rate.
    pawn_prev: HashMap<WorldObjectId, Vec3>,
    pawn_curr: HashMap<WorldObjectId, Vec3>,
    /// `SimClock::total_ticks` at the last `update` that took a snapshot;
    /// snapshots only roll over when this changes, so paused / between-tick
    /// frames keep `prev` and `curr` matched and the lerp pins to live.
    last_seen_tick: u64,

    /// Wall-clock origin for the idle-bob phase (Instant so it ignores sim
    /// speed and pause — the bob keeps breathing while the sim is frozen).
    started: Instant,
    /// Cached bob offset for this frame, recomputed once in `begin_frame`
    /// so multiple pawns drawn in the same frame share one phase.
    frame_bob_offset: f32,
}

impl PawnRenderer {
    pub fn new() -> Self {
        Self {
            pawn_prev: HashMap::new(),
            pawn_curr: HashMap::new(),
            last_seen_tick: 0,
            started: Instant::now(),
            frame_bob_offset: 0.0,
        }
    }

    /// Per-frame snapshot rollover. Previous-tick positions are the lerp
    /// source, this tick's positions the destination. Snapshot only when
    /// the tick counter has advanced — paused frames keep both maps
    /// matched so the lerp pins to the live position.
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

    /// Recompute the per-frame idle-bob offset. Called once by the
    /// top-level view at the start of each render.
    pub fn begin_frame(&mut self) {
        let phase = self.started.elapsed().as_secs_f32() * IDLE_BOB_HZ * TAU;
        self.frame_bob_offset = phase.sin() * IDLE_BOB_AMPLITUDE;
    }

    /// Interpolated world-space position for one pawn this frame: lerp
    /// the two most recent tick-boundary snapshots, then add the idle-bob
    /// offset when the pawn currently has no [`Move`](crate::sim::Move).
    /// The fallback `live_position` covers pawns that haven't been
    /// snapshotted yet (first frame after spawn).
    pub fn interp_position(
        &self,
        id: WorldObjectId,
        live_position: Vec3,
        alpha: f32,
        has_move: bool,
    ) -> Vec3 {
        let prev = self.pawn_prev.get(&id).copied().unwrap_or(live_position);
        let curr = self.pawn_curr.get(&id).copied().unwrap_or(live_position);
        let mut pos = prev.lerp(curr, alpha);
        if !has_move {
            pos.z += self.frame_bob_offset;
        }
        pos
    }
}

/// Pawn body + a small offset cube ("satchel") on the local +X side, baked
/// into one mesh. Same material as the body, so the satchel doesn't pop
/// visually — but the geometric protrusion catches the sun at a
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
