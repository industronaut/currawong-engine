//! Fixed-tick simulation clock.
//!
//! The simulation always sees a constant tick period regardless of speed —
//! varying [`SimClock::set_speed`] only changes how many ticks fire per
//! wall-clock second. This keeps sim logic deterministic at any playback rate.

use std::time::Duration;

/// Maximum simulation ticks consumed per render frame. Caps speed-driven
/// catch-up to prevent spiral-of-death when the sim falls behind real time.
const MAX_TICKS_PER_FRAME: u32 = 16;

/// Drives a fixed-tick simulation from wall-clock time, with speed scaling.
///
/// The simulation always sees a constant [`tick_period`](Self::tick_period)
/// per call to [`Simulation::tick`](crate::Simulation::tick); varying `speed`
/// only changes how many ticks fire per wall-clock second. This keeps sim
/// logic deterministic regardless of playback speed.
///
/// Pause is `speed = 0.0`. Reverse playback (negative speed) is not supported.
pub struct SimClock {
    speed: f32,
    tick_period: Duration,
    accumulator: Duration,
    sim_time: Duration,
    total_ticks: u64,
}

impl SimClock {
    /// Default tick rate when none is specified.
    pub const DEFAULT_TICK_HZ: u32 = 60;

    /// Create a clock at 60 Hz, speed 1.0.
    pub fn new() -> Self {
        Self::with_tick_rate(Self::DEFAULT_TICK_HZ)
    }

    /// Create a clock at the given tick rate, speed 1.0.
    pub fn with_tick_rate(hz: u32) -> Self {
        assert!(hz > 0, "tick rate must be > 0");
        Self {
            speed: 1.0,
            tick_period: Duration::from_secs_f64(1.0 / hz as f64),
            accumulator: Duration::ZERO,
            sim_time: Duration::ZERO,
            total_ticks: 0,
        }
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Set the speed multiplier. `1.0` is real-time; `2.0` is 2x; `0.5` is
    /// half-speed; `0.0` is paused. Negative, NaN, and infinite values are
    /// treated as `0.0` — `advance` with infinite speed would otherwise
    /// overflow the accumulator on the very first tick.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = if speed.is_finite() {
            speed.max(0.0)
        } else {
            0.0
        };
    }

    pub fn is_paused(&self) -> bool {
        self.speed == 0.0
    }

    pub fn tick_period(&self) -> Duration {
        self.tick_period
    }

    /// Number of simulation ticks elapsed since the clock was created.
    pub fn total_ticks(&self) -> u64 {
        self.total_ticks
    }

    /// Total simulated time elapsed (`total_ticks * tick_period`). Differs
    /// from wall time when speed is not 1.0.
    pub fn sim_time(&self) -> Duration {
        self.sim_time
    }

    /// Interpolation factor in `[0, 1]` between the most recent tick and the
    /// next pending tick. Pass to [`View::render`](crate::View::render) for
    /// smooth animation when tick rate is below refresh rate.
    pub fn alpha(&self) -> f32 {
        let acc = self.accumulator.as_secs_f64();
        let period = self.tick_period.as_secs_f64();
        (acc / period).clamp(0.0, 1.0) as f32
    }

    /// Advance the clock by `wall_dt` and return the number of sim ticks the
    /// caller should run. Caps at `MAX_TICKS_PER_FRAME` to prevent
    /// spiral-of-death; remaining accumulator is dropped at the cap.
    pub fn advance(&mut self, wall_dt: Duration) -> u32 {
        if self.speed <= 0.0 {
            return 0;
        }
        self.accumulator += wall_dt.mul_f32(self.speed);
        let mut ticks = 0;
        while self.accumulator >= self.tick_period && ticks < MAX_TICKS_PER_FRAME {
            self.accumulator -= self.tick_period;
            self.sim_time += self.tick_period;
            self.total_ticks += 1;
            ticks += 1;
        }
        if ticks == MAX_TICKS_PER_FRAME {
            self.accumulator = Duration::ZERO;
        }
        ticks
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn period_at(hz: u32) -> Duration {
        Duration::from_secs_f64(1.0 / hz as f64)
    }

    #[test]
    fn new_clock_has_default_state() {
        let clock = SimClock::new();
        assert_eq!(clock.speed(), 1.0);
        assert!(!clock.is_paused());
        assert_eq!(clock.total_ticks(), 0);
        assert_eq!(clock.sim_time(), Duration::ZERO);
        assert_eq!(clock.tick_period(), period_at(SimClock::DEFAULT_TICK_HZ));
        assert_eq!(clock.alpha(), 0.0);
    }

    #[test]
    #[should_panic(expected = "tick rate must be > 0")]
    fn zero_tick_rate_panics() {
        SimClock::with_tick_rate(0);
    }

    #[test]
    fn advance_zero_fires_no_ticks() {
        let mut clock = SimClock::new();
        assert_eq!(clock.advance(Duration::ZERO), 0);
        assert_eq!(clock.total_ticks(), 0);
        assert_eq!(clock.sim_time(), Duration::ZERO);
    }

    #[test]
    fn advance_two_periods_fires_two_ticks() {
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        assert_eq!(clock.advance(period * 2), 2);
        assert_eq!(clock.total_ticks(), 2);
        assert_eq!(clock.sim_time(), period * 2);
    }

    #[test]
    fn sub_period_advances_accumulate_across_calls() {
        // Pins the "don't zero the accumulator on natural loop exit" branch:
        // half a period twice should fire exactly one tick on the second call.
        // 1 Hz keeps period/2 = 0.5s — exact through Duration::mul_f32.
        let mut clock = SimClock::with_tick_rate(1);
        let period = clock.tick_period();
        assert_eq!(clock.advance(period / 2), 0);
        assert!((clock.alpha() - 0.5).abs() < 1e-3);
        assert_eq!(clock.advance(period / 2), 1);
        assert_eq!(clock.total_ticks(), 1);
    }

    #[test]
    fn advance_over_cap_fires_max_and_drops_remainder() {
        // Wall dt is 2x the cap — only MAX_TICKS_PER_FRAME should fire and the
        // leftover accumulator should be dropped (spiral-of-death guard).
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        assert_eq!(
            clock.advance(period * MAX_TICKS_PER_FRAME * 2),
            MAX_TICKS_PER_FRAME
        );
        assert_eq!(clock.alpha(), 0.0);
        // No leftover survives into the next frame.
        assert_eq!(clock.advance(Duration::ZERO), 0);
    }

    #[test]
    fn advance_exactly_at_cap_leaves_accumulator_at_zero() {
        // Boundary: exactly MAX_TICKS_PER_FRAME * tick_period. The loop hits
        // the cap with the accumulator already at zero, so the zeroing branch
        // is a no-op rather than dropping work.
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        assert_eq!(
            clock.advance(period * MAX_TICKS_PER_FRAME),
            MAX_TICKS_PER_FRAME
        );
        assert_eq!(clock.alpha(), 0.0);
    }

    #[test]
    fn pause_fires_no_ticks() {
        let mut clock = SimClock::new();
        clock.set_speed(0.0);
        assert!(clock.is_paused());
        assert_eq!(clock.advance(Duration::from_secs(3600)), 0);
        assert_eq!(clock.total_ticks(), 0);
        assert_eq!(clock.alpha(), 0.0);
    }

    #[test]
    fn double_speed_doubles_effective_tick_rate() {
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        clock.set_speed(2.0);
        // One period of wall time at 2x speed fires two ticks.
        assert_eq!(clock.advance(period), 2);
    }

    #[test]
    fn half_speed_halves_effective_tick_rate() {
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        clock.set_speed(0.5);
        // One period of wall time at half-speed fires zero ticks; two periods
        // of wall time fire one tick.
        assert_eq!(clock.advance(period), 0);
        assert_eq!(clock.advance(period), 1);
    }

    #[test]
    fn tick_period_is_constant_regardless_of_speed() {
        // Determinism guard: changing speed must not change tick_period.
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        clock.set_speed(0.25);
        assert_eq!(clock.tick_period(), period);
        clock.set_speed(8.0);
        assert_eq!(clock.tick_period(), period);
    }

    #[test]
    fn alpha_is_zero_immediately_after_a_tick() {
        // 1 Hz: advance(1s).mul_f32(1.0) round-trips exactly, so the accumulator
        // lands at exactly zero after the tick.
        let mut clock = SimClock::with_tick_rate(1);
        let period = clock.tick_period();
        clock.advance(period);
        assert_eq!(clock.alpha(), 0.0);
    }

    #[test]
    fn alpha_approaches_one_mid_tick_without_exceeding() {
        let mut clock = SimClock::new();
        let period = clock.tick_period();
        clock.advance(period.mul_f32(0.99));
        let alpha = clock.alpha();
        assert!(alpha < 1.0, "alpha {alpha} should be < 1");
        assert!(alpha > 0.98, "alpha {alpha} should be > 0.98");
    }

    #[test]
    fn alpha_clamps_at_one_if_accumulator_overshoots_period() {
        // The tick loop drains the accumulator below tick_period in normal use,
        // so this clamp is a safety net. Stuff the accumulator directly to
        // exercise it.
        let mut clock = SimClock::new();
        clock.accumulator = clock.tick_period * 4;
        assert_eq!(clock.alpha(), 1.0);
    }

    #[test]
    fn negative_speed_is_clamped_to_zero() {
        let mut clock = SimClock::new();
        clock.set_speed(-1.0);
        assert_eq!(clock.speed(), 0.0);
        assert!(clock.is_paused());
    }

    #[test]
    fn nan_speed_is_treated_as_zero() {
        let mut clock = SimClock::new();
        clock.set_speed(f32::NAN);
        assert_eq!(clock.speed(), 0.0);
        assert!(clock.is_paused());
    }

    #[test]
    fn infinite_speed_is_treated_as_zero() {
        let mut clock = SimClock::new();
        clock.set_speed(f32::INFINITY);
        assert_eq!(clock.speed(), 0.0);
        assert!(clock.is_paused());
        clock.set_speed(f32::NEG_INFINITY);
        assert_eq!(clock.speed(), 0.0);
        assert!(clock.is_paused());
    }
}
