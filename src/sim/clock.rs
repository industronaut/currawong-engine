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
    /// half-speed; `0.0` is paused. Negative values are clamped to zero.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.max(0.0);
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
