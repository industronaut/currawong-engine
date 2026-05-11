//! View-side reconciler for declarative emitter attachments.
//!
//! Pairs with the per-frame `extract_visuals` pattern: when extract returns
//! an "attach emitter at this transform" leaf, the user calls
//! [`EmitterReconciler::declare`]. The reconciler lazy-creates new live
//! emitters, updates existing ones, integrates their particles each frame,
//! and GC's emitters whose parent has been removed (or whose attachment was
//! not declared).
//!
//! Frame loop:
//! ```text
//! emitters.begin_frame();
//! for each sim object: extract_visuals(...) {
//!     VisualPart::AttachEmitter { template, slot, transform } =>
//!         emitters.declare(world_ref, slot, template, transform);
//! }
//! emitters.integrate(&sim.zones, dt);
//! for (template, particle) in emitters.iter_particles() { /* render */ }
//! ```
//!
//! Particle visual properties (size, colour, sprite) are *not* engine
//! concerns — the engine only models kinematics and lifecycle. Keep visual
//! params in a parallel user-side table keyed by the same `E` template id.

use std::collections::HashMap;
use std::hash::Hash;

use glam::{Mat4, Vec3};

use crate::sim::{WorldObjectRef, Zones};

/// CPU-side state of one live particle. Read by the user during the render
/// extract to populate per-particle GPU instance data.
#[derive(Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
}

impl Particle {
    /// Lifetime fraction in `[0, 1]`. Useful for interpolating visual
    /// properties (size, colour) from a base value to an end value.
    pub fn life(&self) -> f32 {
        (self.age / self.lifetime.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0)
    }
}

/// What happens to a live emitter when its parent goes away — either the
/// parent's [`WorldObjectRef`] no longer resolves in `Zones`, or the
/// attachment was not declared this frame (which means the user's extract
/// function chose not to re-emit it, e.g. campfire was extinguished).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParticleLifecycle {
    /// Snap dead immediately. Live particles are dropped on the next
    /// integrate. Right for instant cuts (object destroyed, UI hidden).
    Cut,
    /// Stop spawning but let live particles finish their lifetime. Right
    /// for natural fade-outs (campfire extinguished, jet engine cut).
    Linger,
}

/// Static recipe shared across live emitters that reference it. Defines the
/// particle kinematics; visual properties (size, colour, sprite) belong to
/// the user's render code, keyed by the same template id.
#[derive(Clone, Copy, Debug)]
pub struct EmitterTemplate {
    /// Particles spawned per second when the emitter is alive.
    pub spawn_rate: f32,
    /// Lifetime of each particle in seconds.
    pub particle_lifetime: f32,
    /// Initial speed magnitude along the spawn direction.
    pub initial_speed: f32,
    /// `±` range added to `initial_speed` per particle.
    pub speed_jitter: f32,
    /// Mean spawn direction. Will be normalised by the reconciler.
    pub direction: Vec3,
    /// Half-angle (radians) of the cone around `direction` within which
    /// initial velocities are sampled. `0.0` for a beam, `π/2` for a
    /// hemisphere, `π` for a full sphere.
    pub cone_half_angle: f32,
    /// Per-emitter live particle cap. Spawns are dropped when reached.
    pub max_particles: u32,
    /// What to do when the parent goes away.
    pub on_parent_lost: ParticleLifecycle,
}

/// Manages live emitter attachments keyed by `(WorldObjectRef, slot)`.
///
/// Generic over:
/// - `E` — emitter template id (e.g. `enum EmitterId { Flame, Smoke }`)
/// - `S` — slot id, distinguishing multiple emitters on one parent
///   (e.g. `enum EmitterSlot { Flame, Smoke, Sparks }`)
///
/// Particles emit in *world space*: once spawned they evolve independently
/// of the parent's subsequent transform. Local-space particles (which would
/// drag along with a moving parent) are not supported yet — when added, they
/// will be a per-template flag.
pub struct EmitterReconciler<E, S>
where
    E: Copy + Eq + Hash,
    S: Copy + Eq + Hash,
{
    templates: HashMap<E, EmitterTemplate>,
    live: HashMap<(WorldObjectRef, S), LiveEmitter<E>>,
    rng: Rng,
}

struct LiveEmitter<E> {
    template_id: E,
    transform: Mat4,
    declared_this_frame: bool,
    /// Once true, no new particles spawn. Cut-mode emitters are dropped on
    /// the next integrate; Linger emitters keep ticking until empty.
    extinguished: bool,
    spawn_accumulator: f32,
    particles: Vec<Particle>,
}

impl<E, S> EmitterReconciler<E, S>
where
    E: Copy + Eq + Hash,
    S: Copy + Eq + Hash,
{
    /// Create an empty reconciler. `seed` controls the spawn-direction RNG;
    /// pass a fixed value for reproducible visuals across runs.
    pub fn new(seed: u64) -> Self {
        Self {
            templates: HashMap::new(),
            live: HashMap::new(),
            rng: Rng::new(seed),
        }
    }

    /// Register a template under `id`. Re-registering replaces the existing
    /// template; live emitters using it pick up the new parameters on the
    /// next spawn (existing particles are unaffected).
    pub fn register_template(&mut self, id: E, template: EmitterTemplate) {
        self.templates.insert(id, template);
    }

    /// Total live particle count across all emitters. Useful for sizing
    /// instance buckets and for debug overlays.
    pub fn live_particle_count(&self) -> usize {
        self.live.values().map(|e| e.particles.len()).sum()
    }

    /// Number of currently-tracked live emitters (declared and lingering).
    pub fn live_emitter_count(&self) -> usize {
        self.live.len()
    }

    /// Mark all live emitters as not-declared. Call once at the start of
    /// each frame, before iterating sim and declaring this frame's
    /// attachments.
    pub fn begin_frame(&mut self) {
        for emitter in self.live.values_mut() {
            emitter.declared_this_frame = false;
        }
    }

    /// Declare an emitter attachment on `parent` at `slot`, using `template`,
    /// at world transform `transform`. Lazy-creates the live emitter if one
    /// doesn't exist for this `(parent, slot)` key; otherwise updates the
    /// existing one's transform and clears its extinguished state.
    ///
    /// Panics if `template` was not registered via
    /// [`register_template`](Self::register_template) — that's a programming
    /// error, not a runtime condition.
    pub fn declare(&mut self, parent: WorldObjectRef, slot: S, template: E, transform: Mat4) {
        assert!(
            self.templates.contains_key(&template),
            "EmitterReconciler::declare with unregistered template — register at init"
        );
        let key = (parent, slot);
        let emitter = self.live.entry(key).or_insert_with(|| LiveEmitter {
            template_id: template,
            transform,
            declared_this_frame: true,
            extinguished: false,
            spawn_accumulator: 0.0,
            particles: Vec::new(),
        });
        emitter.template_id = template;
        emitter.transform = transform;
        emitter.declared_this_frame = true;
        emitter.extinguished = false;
    }

    /// Tick all live emitters by `dt` seconds. Spawns new particles based
    /// on each template's rate, advances existing ones, drops dead ones,
    /// and GCs emitters that have been extinguished and have no live
    /// particles left.
    ///
    /// `zones` is consulted to detect parents whose `WorldObjectRef` no
    /// longer resolves — those emitters are extinguished according to their
    /// template's [`ParticleLifecycle`].
    pub fn integrate(&mut self, zones: &Zones, dt: f32) {
        let templates = &self.templates;
        let rng = &mut self.rng;
        self.live.retain(|key, emitter| {
            let template = templates
                .get(&emitter.template_id)
                .expect("emitter references a registered template");

            // If parent gone or attachment not declared this frame, the
            // emitter is no longer "alive" for spawn purposes.
            let parent_alive = key.0.resolve(zones).is_some();
            if !emitter.declared_this_frame || !parent_alive {
                emitter.extinguished = true;
            }

            // Cut-mode emitters drop their particles on extinguish.
            if emitter.extinguished && template.on_parent_lost == ParticleLifecycle::Cut {
                return false;
            }

            // Tick existing particles.
            for p in emitter.particles.iter_mut() {
                p.age += dt;
                p.position += p.velocity * dt;
            }
            emitter.particles.retain(|p| p.age < p.lifetime);

            // Spawn new particles unless extinguished.
            if !emitter.extinguished {
                emitter.spawn_accumulator += template.spawn_rate * dt;
                while emitter.spawn_accumulator >= 1.0
                    && (emitter.particles.len() as u32) < template.max_particles
                {
                    emitter.spawn_accumulator -= 1.0;
                    emitter
                        .particles
                        .push(spawn_particle(template, emitter.transform, rng));
                }
            } else {
                emitter.spawn_accumulator = 0.0;
            }

            // Keep the emitter if it might still spawn or has live particles.
            !emitter.extinguished || !emitter.particles.is_empty()
        });
    }

    /// Iterate `(template_id, &particle)` pairs across all live particles.
    /// Use this to drive the user's particle render path: bucket by
    /// template id, look up visual params keyed by the same id, push GPU
    /// instance data.
    ///
    /// Order is unspecified.
    pub fn iter_particles(&self) -> impl Iterator<Item = (E, &Particle)> + '_ {
        self.live
            .values()
            .flat_map(|e| e.particles.iter().map(move |p| (e.template_id, p)))
    }
}

fn spawn_particle(template: &EmitterTemplate, transform: Mat4, rng: &mut Rng) -> Particle {
    let dir = jitter_direction(template.direction, template.cone_half_angle, rng);
    let speed = (template.initial_speed + rng.bipolar() * template.speed_jitter).max(0.0);
    Particle {
        position: transform.transform_point3(Vec3::ZERO),
        velocity: dir * speed,
        age: 0.0,
        lifetime: template.particle_lifetime,
    }
}

/// Random unit vector within `half_angle` radians of `axis`. `axis` is
/// expected to be (approximately) normalised; this re-normalises the result.
fn jitter_direction(axis: Vec3, half_angle: f32, rng: &mut Rng) -> Vec3 {
    let axis = axis.normalize_or(Vec3::Z);
    if half_angle <= 0.0 {
        return axis;
    }
    let helper = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let tangent = axis.cross(helper).normalize();
    let bitangent = axis.cross(tangent);
    let theta = rng.unit() * half_angle;
    let phi = rng.unit() * std::f32::consts::TAU;
    let st = theta.sin();
    let ct = theta.cos();
    (axis * ct + tangent * (st * phi.cos()) + bitangent * (st * phi.sin())).normalize()
}

// --- Tiny xorshift RNG ---------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / u64::MAX as f64) as f32
    }
    fn bipolar(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Zone, ZoneId, Zones};

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum E {
        Flame,
    }
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum S {
        Main,
    }

    fn linger_template(spawn_rate: f32, lifetime: f32) -> EmitterTemplate {
        EmitterTemplate {
            spawn_rate,
            particle_lifetime: lifetime,
            initial_speed: 1.0,
            speed_jitter: 0.0,
            direction: Vec3::Z,
            cone_half_angle: 0.0,
            max_particles: 64,
            on_parent_lost: ParticleLifecycle::Linger,
        }
    }

    fn cut_template() -> EmitterTemplate {
        EmitterTemplate {
            on_parent_lost: ParticleLifecycle::Cut,
            ..linger_template(60.0, 1.0)
        }
    }

    fn one_object_zone() -> (Zones, WorldObjectRef) {
        let mut zones = Zones::new();
        let zid: ZoneId = zones.insert(Zone::new());
        let oid = zones.get_mut(zid).unwrap().insert(Default::default());
        (zones, WorldObjectRef { zone: zid, id: oid })
    }

    #[test]
    fn declare_creates_live_emitter() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(60.0, 1.0));
        let (_zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);

        assert_eq!(r.live_emitter_count(), 1);
    }

    #[test]
    fn integrate_spawns_particles_at_rate() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(100.0, 10.0));
        let (zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.1); // 100 hz × 0.1 s = 10 particles

        assert_eq!(r.live_particle_count(), 10);
    }

    #[test]
    fn particles_die_when_age_exceeds_lifetime() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(10.0, 0.5));
        let (zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.1);
        assert!(r.live_particle_count() > 0);

        // Stop spawning so we observe pure decay.
        r.begin_frame(); // no declare → extinguished
        r.integrate(&zones, 1.0);
        assert_eq!(r.live_particle_count(), 0);
    }

    #[test]
    fn linger_keeps_particles_after_undeclared() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(100.0, 1.0));
        let (zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.1);
        let n = r.live_particle_count();
        assert!(n > 0);

        // Undeclare; particles should still be alive briefly.
        r.begin_frame();
        r.integrate(&zones, 0.05);
        assert_eq!(r.live_particle_count(), n);
        assert_eq!(
            r.live_emitter_count(),
            1,
            "emitter lingers while particles alive"
        );
    }

    #[test]
    fn cut_drops_particles_when_undeclared() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, cut_template());
        let (zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.1);
        assert!(r.live_particle_count() > 0);

        r.begin_frame(); // no declare
        r.integrate(&zones, 0.0);
        assert_eq!(r.live_particle_count(), 0);
        assert_eq!(
            r.live_emitter_count(),
            0,
            "cut emitter is dropped immediately"
        );
    }

    #[test]
    fn parent_removal_extinguishes_emitter() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(100.0, 0.2));
        let (mut zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.05);
        assert!(r.live_particle_count() > 0);

        // Remove parent. Even though we re-declare (which we wouldn't in
        // practice), parent absence should still extinguish.
        zones.get_mut(parent.zone).unwrap().remove(parent.id);
        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 1.0);
        assert_eq!(r.live_emitter_count(), 0);
    }

    #[test]
    fn redeclare_updates_transform_without_resetting_particles() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        r.register_template(E::Flame, linger_template(100.0, 5.0));
        let (zones, parent) = one_object_zone();

        r.begin_frame();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
        r.integrate(&zones, 0.1);
        let n = r.live_particle_count();
        assert!(n > 0);

        // Re-declare with a different transform; existing particles should
        // be preserved (they're in world space already).
        r.begin_frame();
        r.declare(
            parent,
            S::Main,
            E::Flame,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );
        r.integrate(&zones, 0.0); // dt=0 so no new spawns
        assert_eq!(r.live_particle_count(), n);
    }

    #[test]
    #[should_panic(expected = "unregistered template")]
    fn declare_unregistered_template_panics() {
        let mut r = EmitterReconciler::<E, S>::new(1);
        let (_zones, parent) = one_object_zone();
        r.declare(parent, S::Main, E::Flame, Mat4::IDENTITY);
    }
}
