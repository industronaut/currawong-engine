//! Sim-side environment state — time of day, day count, and the helper that
//! turns time-of-day into a sun direction.
//!
//! [`SimEnvironment`] is a helper struct, not a required trait: the user
//! holds one on their `Simulation` impl and exposes it however they like
//! (mirrors how [`Camera`](crate::Camera) is opt-in on the view side).
//!
//! The split is the same as the rest of the engine — sim owns the *facts*
//! (it's 14:30 on day 47), the view owns the *appearance* (sun at this
//! angle, this colour). The view reads this struct each frame and produces
//! a `ViewEnvironment` for the GPU.
//!
//! ## Determinism
//!
//! State is integer-backed: [`time_of_day`](SimEnvironment::time_of_day) is
//! Q16.16 in `[0, 1)`, [`seconds_per_day`](SimEnvironment::seconds_per_day)
//! is Q16.16 seconds, and [`Self::advance`] does its math in integer
//! nanoseconds before converting to fractional seconds. The float boundary
//! is one-way — [`sun_direction_for`] converts time-of-day to a sun
//! direction via `f32` `sin`/`cos`, which is acceptable because the
//! output flows into the view and is not read back into sim logic.

use std::time::Duration;

use glam::Vec3;

use super::units::SimUnit;

/// Sim-side environment state. Time of day, day count, and (later) weather
/// and season. View-agnostic — held by the user's `Simulation` impl,
/// advanced by their `tick`, and read each frame by the view's
/// `extract_environment`.
#[derive(Clone, Copy, Debug)]
pub struct SimEnvironment {
    /// Day fraction in `[0, 1)`: `0` is midnight, `0.5` is noon. Q16.16,
    /// so resolution is ~1.3s per 24-hour day — finer than any visual
    /// effect cares about.
    pub time_of_day: SimUnit,
    /// Days since the world was created. Increments when `time_of_day`
    /// wraps past `1.0`.
    pub day: u32,
    /// How many real seconds one in-world day takes when sim speed is 1.0.
    /// Default 600s (10 minutes per day) — short enough to watch the sun
    /// cross the sky during a play session.
    pub seconds_per_day: SimUnit,
}

impl SimEnvironment {
    /// Default: starts at sunrise (`time_of_day = 0.25`) on day 0.
    pub fn new() -> Self {
        Self {
            time_of_day: SimUnit::from_num(0.25),
            day: 0,
            seconds_per_day: SimUnit::from_num(600),
        }
    }

    /// Advance time-of-day by `dt`. Converts the `Duration` to fractional
    /// seconds via integer nanoseconds (bit-exact), then divides by
    /// [`Self::seconds_per_day`] to get the fractional-day delta. Wraps
    /// [`Self::time_of_day`] and increments [`Self::day`] at midnight.
    pub fn advance(&mut self, dt: Duration) {
        if self.seconds_per_day <= SimUnit::ZERO {
            return;
        }
        // Q16.16 seconds from integer nanoseconds. u128 keeps the
        // shifted intermediate from overflowing for long durations.
        let nanos = dt.as_nanos();
        let secs_q16: u128 = (nanos << 16) / 1_000_000_000;
        let secs_q16_i32 = secs_q16.min(i32::MAX as u128) as i32;
        let dt_secs = SimUnit::from_bits(secs_q16_i32);
        self.time_of_day += dt_secs / self.seconds_per_day;
        while self.time_of_day >= SimUnit::from_num(1) {
            self.time_of_day -= SimUnit::from_num(1);
            self.day = self.day.saturating_add(1);
        }
    }
}

impl Default for SimEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Trivial sun-direction model: midnight at `0` (sun under -Z), sunrise at
/// `0.25` (sun at +X, eastern horizon), noon at `0.5` (sun overhead +Z),
/// sunset at `0.75` (sun at -X). Returns a unit vector pointing *from the
/// world toward the sun*, in the engine's right-handed Z-up frame. The arc
/// is east-to-west through the +Y meridian — equator at the equinox, no
/// axial tilt or latitude.
///
/// Good enough for prototyping. Replace when you care about latitude,
/// season, or axial tilt.
///
/// Takes a [`SimUnit`] for symmetry with the rest of sim state; the
/// conversion to `f32` for the trig happens internally. This is the
/// "sim-internal float trig that produces a `Vec3` for the view" the
/// determinism issue calls out — fine in single-arch builds, would need
/// a fixed-point sin/cos LUT before cross-arch lockstep multiplayer.
pub fn sun_direction_for(time_of_day: SimUnit) -> Vec3 {
    let frac = time_of_day.to_num::<f32>().rem_euclid(1.0);
    let theta = frac * std::f32::consts::TAU;
    Vec3::new(theta.sin(), 0.0, -theta.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    fn t(v: f32) -> SimUnit {
        SimUnit::from_num(v)
    }

    #[test]
    fn sun_direction_cardinal_points() {
        assert!(approx(sun_direction_for(t(0.00)), -Vec3::Z)); // midnight
        assert!(approx(sun_direction_for(t(0.25)), Vec3::X)); // sunrise (east)
        assert!(approx(sun_direction_for(t(0.50)), Vec3::Z)); // noon (overhead)
        assert!(approx(sun_direction_for(t(0.75)), -Vec3::X)); // sunset (west)
    }

    #[test]
    fn sun_direction_wraps() {
        assert!(approx(sun_direction_for(t(1.0)), sun_direction_for(t(0.0))));
        assert!(approx(
            sun_direction_for(t(1.25)),
            sun_direction_for(t(0.25))
        ));
        // Negative inputs also wrap (rem_euclid semantics).
        assert!(approx(
            sun_direction_for(t(-0.25)),
            sun_direction_for(t(0.75))
        ));
    }

    #[test]
    fn advance_wraps_day() {
        let mut env = SimEnvironment {
            time_of_day: t(0.9),
            day: 4,
            seconds_per_day: t(100.0),
        };
        env.advance(Duration::from_secs(20)); // +0.2 → wraps to 0.1, day += 1
        let diff = (env.time_of_day - t(0.1)).abs();
        // Within Q16.16 quantization (a few bits across two ops).
        assert!(diff <= SimUnit::from_bits(8), "drift {diff}");
        assert_eq!(env.day, 5);
    }

    #[test]
    fn advance_zero_period_is_noop() {
        let mut env = SimEnvironment {
            time_of_day: t(0.3),
            day: 0,
            seconds_per_day: SimUnit::ZERO,
        };
        env.advance(Duration::from_secs(100));
        assert_eq!(env.time_of_day, t(0.3));
        assert_eq!(env.day, 0);
    }

    #[test]
    fn advance_is_bit_exact_across_repeats() {
        // Determinism: replay the same sequence of advance() calls from
        // the same starting state and assert the result is identical.
        fn run() -> (SimUnit, u32) {
            let mut env = SimEnvironment::new();
            for _ in 0..3600 {
                env.advance(Duration::from_millis(16));
            }
            (env.time_of_day, env.day)
        }
        assert_eq!(run(), run());
    }
}
