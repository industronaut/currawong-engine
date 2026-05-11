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

use glam::Vec3;

/// Sim-side environment state. Time of day, day count, and (later) weather
/// and season. View-agnostic — held by the user's `Simulation` impl,
/// advanced by their `tick`, and read each frame by the view's
/// `extract_environment`.
#[derive(Clone, Copy, Debug)]
pub struct SimEnvironment {
    /// Day fraction in `[0.0, 1.0)`: `0.0` is midnight, `0.5` is noon.
    pub time_of_day: f32,
    /// Days since the world was created. Increments when `time_of_day`
    /// wraps past `1.0`.
    pub day: u32,
    /// How many real seconds one in-world day takes when sim speed is 1.0.
    /// Default 600s (10 minutes per day) — short enough to watch the sun
    /// cross the sky during a play session.
    pub seconds_per_day: f32,
}

impl SimEnvironment {
    /// Default: starts at sunrise (`time_of_day = 0.25`) on day 0.
    pub fn new() -> Self {
        Self {
            time_of_day: 0.25,
            day: 0,
            seconds_per_day: 600.0,
        }
    }

    /// Advance time-of-day by a tick. `dt_seconds` is the wall-clock
    /// equivalent of one sim tick (`SimClock::tick_period` as seconds).
    /// Wraps `time_of_day` and increments `day` at midnight.
    pub fn advance(&mut self, dt_seconds: f32) {
        if self.seconds_per_day <= 0.0 {
            return;
        }
        self.time_of_day += dt_seconds / self.seconds_per_day;
        while self.time_of_day >= 1.0 {
            self.time_of_day -= 1.0;
            self.day = self.day.saturating_add(1);
        }
    }
}

impl Default for SimEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Trivial sun-direction model: midnight at `0.0` (sun under -Z), sunrise at
/// `0.25` (sun at +X, eastern horizon), noon at `0.5` (sun overhead +Z),
/// sunset at `0.75` (sun at -X). Returns a unit vector pointing *from the
/// world toward the sun*, in the engine's right-handed Z-up frame. The arc
/// is east-to-west through the +Y meridian — equator at the equinox, no
/// axial tilt or latitude.
///
/// Good enough for prototyping. Replace when you care about latitude,
/// season, or axial tilt.
pub fn sun_direction_for(time_of_day: f32) -> Vec3 {
    let theta = time_of_day.rem_euclid(1.0) * std::f32::consts::TAU;
    Vec3::new(theta.sin(), 0.0, -theta.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn sun_direction_cardinal_points() {
        assert!(approx(sun_direction_for(0.00), -Vec3::Z)); // midnight
        assert!(approx(sun_direction_for(0.25), Vec3::X)); // sunrise (east)
        assert!(approx(sun_direction_for(0.50), Vec3::Z)); // noon (overhead)
        assert!(approx(sun_direction_for(0.75), -Vec3::X)); // sunset (west)
    }

    #[test]
    fn sun_direction_wraps() {
        assert!(approx(sun_direction_for(1.0), sun_direction_for(0.0)));
        assert!(approx(sun_direction_for(1.25), sun_direction_for(0.25)));
        // Negative inputs also wrap (rem_euclid semantics).
        assert!(approx(sun_direction_for(-0.25), sun_direction_for(0.75)));
    }

    #[test]
    fn advance_wraps_day() {
        let mut env = SimEnvironment {
            time_of_day: 0.9,
            day: 4,
            seconds_per_day: 100.0,
        };
        env.advance(20.0); // +0.2 → wraps to 0.1, day += 1
        assert!((env.time_of_day - 0.1).abs() < 1e-5);
        assert_eq!(env.day, 5);
    }

    #[test]
    fn advance_zero_period_is_noop() {
        let mut env = SimEnvironment {
            time_of_day: 0.3,
            day: 0,
            seconds_per_day: 0.0,
        };
        env.advance(100.0);
        assert_eq!(env.time_of_day, 0.3);
        assert_eq!(env.day, 0);
    }
}
