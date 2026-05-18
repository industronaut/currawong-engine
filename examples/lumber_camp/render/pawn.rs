//! View-side helpers for pawns.
//!
//! The pawn's *body* mesh + texture come from each pawn-kind's RON def
//! via [`super::build_body_template`] — today there's only
//! `currawong:lumberjack`, but additional worker kinds (foreman,
//! apprentice) drop straight in by adding more RON files with
//! `render.shape: "pawn"`. The *carried log* is a shared procedural brown
//! cylinder built once at init and reused across every pawn-kind's
//! template via [`super::PartKey::CarriedLog`]. Log visibility is a
//! view-side decision set by [`update_instance`] from the sim's
//! [`Carrying`] component.
//!
//! View-side pawn state lives on [`PawnRenderer`]: tick-boundary position
//! snapshots for sub-tick interpolation, and the wall-clock idle-bob
//! phase. Both are read by the per-instance update closure in [`super`],
//! which overwrites `instance.world_xform` with the interpolated +
//! bobbed pose; the engine then composes `world_xform *
//! log_local_transform` for the log part automatically.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::time::Instant;

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Components, EngineCtx, LiveRenderObject, PbrMaterial, PrimitiveMesh,
    Renderer, SamplerRegistry, WorldObjectId, WorldObjectRef,
};

use super::{InlineTemplate, MeshTemplate, new_inline_template};
use crate::sim::{Carrying, Game, Move};

/// Index of the carried-log mesh part in every pawn's render template —
/// second part after the body. Stable because
/// [`super::RenderShape::register_template`] adds parts in declaration
/// order and only the pawn shape adds the log.
const LOG_PART: usize = 1;

/// Vertical amplitude of the idle bob applied to pawns without a
/// [`Move`](crate::sim::Move). A few centimetres reads as breathing
/// without looking like a glitch.
const IDLE_BOB_AMPLITUDE: f32 = 0.035;
/// Idle-bob frequency in Hz. Slow enough to feel like a breath rather
/// than a hop; wall-clock driven so paused pawns still breathe.
const IDLE_BOB_HZ: f32 = 1.4;

/// Build the shared carried-log template — a wood-brown horizontal
/// cylinder. Procedural because the carried log is a "you have an
/// inventory item" HUD indicator, not authored content.
pub fn new_log_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
) -> MeshTemplate {
    new_inline_template(
        renderer,
        material,
        samplers,
        asset_server,
        InlineTemplate {
            label: "lumber-camp carried log",
            mesh: &PrimitiveMesh::cylinder(0.07, 0.6, 12, true),
            bounds: Aabb::new(Vec3::new(-0.30, -0.07, -0.07), Vec3::new(0.30, 0.07, 0.07)),
            albedo_factor: Vec4::new(0.42, 0.27, 0.16, 1.0), // wood brown
            metallic: 0.0,
            roughness: 0.85,
        },
    )
}

/// Local-frame transform for the carried log, given the pawn's *body*
/// bounds. Lays the cylinder on its side at roughly shoulder height (~75%
/// of body height — works for whatever-shaped pawns the kind def
/// describes without needing a per-kind shoulder constant in the RON).
pub fn log_local_transform(body_bounds: &Aabb) -> Mat4 {
    let shoulder_z = body_bounds.min.z + (body_bounds.max.z - body_bounds.min.z) * 0.75;
    Mat4::from_rotation_translation(
        Quat::from_rotation_x(PI / 2.0),
        Vec3::new(0.0, 0.0, shoulder_z),
    )
}

/// Visual AABB for a pawn kind's [`RenderTemplate`](currawong::RenderTemplate),
/// extending the body's bounds horizontally to enclose the carried log so
/// the engine frustum-cull doesn't pop the pawn when only its log is
/// on-screen. The log lays along local X, so X gets the extension.
pub fn extended_bounds(body_bounds: &Aabb) -> Aabb {
    let log_half_length = 0.30;
    Aabb::new(
        Vec3::new(
            body_bounds.min.x.min(-log_half_length),
            body_bounds.min.y,
            body_bounds.min.z,
        ),
        Vec3::new(
            body_bounds.max.x.max(log_half_length),
            body_bounds.max.y,
            body_bounds.max.z,
        ),
    )
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
        let lumberjack = &sim.stats.kinds.lumberjack;
        if let Some(zone) = sim.zones.get(sim.zone) {
            for (id, transform) in zone.iter() {
                if zone.components().get::<currawong::data::KindId>(id) == Some(lumberjack) {
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

/// Per-instance update for a live pawn proxy. Called by the dispatcher
/// in [`super`] from inside
/// [`RenderObjectPass::update_instances`](currawong::RenderObjectPass::update_instances).
///
/// Owns *every* sim→view decision for pawns:
/// - overwrites `instance.world_xform` with the interpolated + idle-bobbed
///   pose (engine then composes the carried log's local transform on top
///   for free — the log inherits the interpolated pose automatically);
/// - gates the carried-log part's visibility on the [`Carrying`] component.
pub fn update_instance(
    parent: WorldObjectRef,
    components: &Components,
    instance: &mut LiveRenderObject,
    alpha: f32,
    state: &PawnRenderer,
) {
    let live_position = instance.world_xform.w_axis.truncate();
    let has_move = components.get::<Move>(parent.id).is_some();
    let pos = state.interp_position(parent.id, live_position, alpha, has_move);
    instance.world_xform.w_axis = Vec4::new(pos.x, pos.y, pos.z, 1.0);
    instance.mesh_parts[LOG_PART].visible = components.get::<Carrying>(parent.id).is_some();
}
