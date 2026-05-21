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
//! snapshots for sub-tick interpolation. The idle-bob phase is per-pawn,
//! driven by the sim's [`Idle::seconds`](crate::sim::Idle) counter, so each
//! pawn breathes out of phase with the others (and the bob freezes on
//! pause along with the sim). Read by the per-instance update closure in
//! [`super`], which overwrites `instance.world_xform` with the
//! interpolated + bobbed pose; the engine then composes `world_xform *
//! log_local_transform` for the log part automatically.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

use currawong::glam::{Mat4, Quat, Vec3, Vec4};
use currawong::{
    Aabb, AssetServer, Components, EngineCtx, InlineTemplate, LiveRenderObject, MeshTemplate,
    PbrMaterial, PbrMaterialInstance, PrimitiveMesh, Renderer, SamplerRegistry, WorldObjectId,
    WorldObjectRef,
};

use crate::sim::{Carrying, Game, Idle};

/// Index of the carried-log mesh part in every pawn's render template —
/// second part after the body. Stable because
/// [`super::RenderShape::register_template`] adds parts in declaration
/// order and only the pawn shape adds the log.
const LOG_PART: usize = 1;

/// Vertical amplitude of the idle bob applied to pawns carrying an
/// [`Idle`](crate::sim::Idle) component. A few centimetres reads as
/// breathing without looking like a glitch.
const IDLE_BOB_AMPLITUDE: f32 = 0.035;
/// Idle-bob frequency in Hz. Slow enough to feel like a breath rather
/// than a hop.
const IDLE_BOB_HZ: f32 = 1.4;

/// Build the shared carried-log template — a wood-brown horizontal
/// cylinder. Procedural because the carried log is a "you have an
/// inventory item" HUD indicator, not authored content.
pub fn new_log_template(
    renderer: &Renderer,
    material: &PbrMaterial,
    samplers: &SamplerRegistry,
    asset_server: &AssetServer,
) -> MeshTemplate<PbrMaterialInstance> {
    material.inline_template(
        renderer,
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

/// Vertical air gap between the pawn body's top and the bottom of the
/// carried log. Small enough to read as "balanced on top of the head",
/// large enough to clear the body's hemispherical cap at any pawn-kind's
/// declared bounds.
const LOG_GAP: f32 = 0.08;
/// Radius of the carried-log cylinder. Used to centre the log on its
/// long axis above the head — the cylinder mesh is built around its own
/// origin, so we lift by `radius + gap` to seat it cleanly. Kept in sync
/// with [`new_log_template`]'s `PrimitiveMesh::cylinder(0.07, ...)` call.
const LOG_RADIUS: f32 = 0.07;
/// Half the cylinder's `height` parameter, i.e. how far the log extends
/// along its long axis from its centre. Same kept-in-sync caveat as
/// [`LOG_RADIUS`].
const LOG_HALF_LENGTH: f32 = 0.30;

/// Local-frame transform for the carried log, given the pawn's *body*
/// bounds. Lays the cylinder on its side just above the pawn's head with
/// a small gap, so the log reads clearly without hiding inside the body
/// (the previous "shoulder height" placement put the log entirely inside
/// the body capsule — the cylinder's half-length was equal to the body's
/// radius, so its tips sat right at the body's surface).
pub fn log_local_transform(body_bounds: &Aabb) -> Mat4 {
    let log_centre_z = body_bounds.max.z + LOG_GAP + LOG_RADIUS;
    Mat4::from_rotation_translation(
        Quat::from_rotation_x(PI / 2.0),
        Vec3::new(0.0, 0.0, log_centre_z),
    )
}

/// Visual AABB for a pawn kind's [`RenderTemplate`](currawong::RenderTemplate),
/// extending the body's bounds to enclose the carried log so the engine
/// frustum-cull doesn't pop the pawn when only its log is on-screen.
///
/// After the `Quat::from_rotation_x(PI/2)` in [`log_local_transform`], the
/// cylinder's long axis sits along the pawn's local Y, so Y is the axis
/// that needs the half-length extension; Z gets the head-clearance lift.
pub fn extended_bounds(body_bounds: &Aabb) -> Aabb {
    let log_top_z = body_bounds.max.z + LOG_GAP + 2.0 * LOG_RADIUS;
    Aabb::new(
        Vec3::new(
            body_bounds.min.x,
            body_bounds.min.y.min(-LOG_HALF_LENGTH),
            body_bounds.min.z,
        ),
        Vec3::new(
            body_bounds.max.x,
            body_bounds.max.y.max(LOG_HALF_LENGTH),
            log_top_z,
        ),
    )
}

/// Per-frame pawn-only view state: tick-boundary position snapshots for
/// sub-tick interpolation. Each pawn's idle-bob phase is read per-call
/// from its sim [`Idle`](crate::sim::Idle) counter, so there is no
/// shared per-frame phase state to keep here.
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
}

impl PawnRenderer {
    pub fn new() -> Self {
        Self {
            pawn_prev: HashMap::new(),
            pawn_curr: HashMap::new(),
            last_seen_tick: 0,
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
                    self.pawn_curr.insert(id, transform.position.to_vec3());
                }
            }
        }
        self.last_seen_tick = now_tick;
    }

    /// Interpolated world-space position for one pawn this frame: lerp
    /// the two most recent tick-boundary snapshots, then add an idle-bob
    /// offset whose phase comes from `idle_seconds` (the pawn's sim-side
    /// [`Idle`](crate::sim::Idle) counter). `None` means the pawn has a
    /// job and shouldn't bob. The fallback `live_position` covers pawns
    /// that haven't been snapshotted yet (first frame after spawn).
    pub fn interp_position(
        &self,
        id: WorldObjectId,
        live_position: Vec3,
        alpha: f32,
        idle_seconds: Option<f32>,
    ) -> Vec3 {
        let prev = self.pawn_prev.get(&id).copied().unwrap_or(live_position);
        let curr = self.pawn_curr.get(&id).copied().unwrap_or(live_position);
        let mut pos = prev.lerp(curr, alpha);
        if let Some(seconds) = idle_seconds {
            let phase = seconds * IDLE_BOB_HZ * TAU;
            pos.z += phase.sin() * IDLE_BOB_AMPLITUDE;
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
    let idle_seconds = components
        .get::<Idle>(parent.id)
        .map(|i| i.seconds.to_num::<f32>());
    let pos = state.interp_position(parent.id, live_position, alpha, idle_seconds);
    instance.world_xform.w_axis = Vec4::new(pos.x, pos.y, pos.z, 1.0);
    instance.mesh_parts[LOG_PART].visible = components.get::<Carrying>(parent.id).is_some();
}
