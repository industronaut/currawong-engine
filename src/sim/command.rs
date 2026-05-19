//! [`CommandQueue`] — the single seam through which external mutations reach
//! the simulation.
//!
//! Views (input handlers, UI widgets, scripts) never hold `&mut Sim`. They
//! push values of the sim's [`Simulation::Command`](crate::Simulation::Command)
//! type into a `CommandQueue` instead, and the engine drains the queue at tick
//! boundaries, calling [`Simulation::apply_command`](crate::Simulation::apply_command)
//! once per command.
//!
//! Reifying mutations like this is a precondition for replay, deterministic
//! testing, save-as-command-log, undo, scripted tests, and (later) lockstep
//! multiplayer. Prior art: Quake `usercmd_t`, RTS `Command`/`Order` (StarCraft,
//! AoE), Factorio `InputAction`, RimWorld Multiplayer mod sync methods. All
//! the same shape: a closed enumerable list of intents, one apply seam,
//! deterministic application.
//!
//! ## Tick gating
//!
//! Every queued command carries an `apply_at_tick: u64`. At sim tick `N` the
//! engine drains every command with `apply_at_tick <= N`, in insertion order,
//! and leaves later-scheduled commands in place. Single-player code uses
//! [`push_now`](CommandQueue::push_now), which stamps the current tick so the
//! command applies on the next tick boundary; the explicit
//! [`push`](CommandQueue::push) is for code that schedules into the future
//! (the future multiplayer path delays commands by `K` ticks to absorb
//! network latency).
//!
//! ## Posture
//!
//! Commands must be primitive data — IDs (`WorldObjectId`, `KindId`,
//! `ZoneId`), numbers, enums, owned strings. Anything carrying an
//! `Arc<dyn Trait>`, a closure, or a borrowed reference has gone wrong: it
//! breaks the future serialise-for-wire / serialise-for-replay path. If a
//! command needs lookup data, it carries the id and `apply_command` resolves
//! it against current sim state at apply time.

/// Buffer between view-side intent and sim-side mutation.
///
/// Generic over the sim's command type (`<S as Simulation>::Command`). Views
/// push commands during `input` / `update` / `ui` / `game_ui`; the engine
/// drains commands whose `apply_at_tick <= current_tick` at each sim tick
/// boundary, applying them via [`Simulation::apply_command`](crate::Simulation::apply_command)
/// before [`Simulation::tick`](crate::Simulation::tick) runs.
///
/// `current_tick` is updated by the engine at frame start so
/// [`push_now`](Self::push_now) can stamp without a clock lookup at every
/// call site.
pub struct CommandQueue<C> {
    pending: Vec<PendingCommand<C>>,
    current_tick: u64,
}

// Fields are read by `drain_ready`, which is `pub(crate)` and unused in the
// sim-only build. Silencing dead_code here keeps `--no-default-features`
// warning-clean without hiding real dead code in the public API.
#[allow(dead_code)]
struct PendingCommand<C> {
    apply_at_tick: u64,
    command: C,
}

impl<C> CommandQueue<C> {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            current_tick: 0,
        }
    }

    /// Push a command to be applied at `apply_at_tick`. If the tick has
    /// already passed, the command drains on the next tick boundary.
    pub fn push(&mut self, apply_at_tick: u64, command: C) {
        self.pending.push(PendingCommand {
            apply_at_tick,
            command,
        });
    }

    /// Push a command to be applied at the current sim tick. The
    /// convenience form — most single-player code wants this.
    pub fn push_now(&mut self, command: C) {
        self.push(self.current_tick, command);
    }

    /// Sim tick the queue believes is current. Set by the engine; useful
    /// when a caller wants to schedule a command `K` ticks ahead with
    /// `push(queue.current_tick() + K, cmd)`.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Number of queued commands (ready + future-scheduled).
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Engine seam: update `current_tick` so subsequent [`push_now`] calls
    /// stamp with the new value.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn set_current_tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Engine seam: pop every command with `apply_at_tick <= current_tick`,
    /// in insertion order, leaving later-scheduled commands in place. The
    /// closure is invoked once per ready command — the typical caller hands
    /// each one straight to [`Simulation::apply_command`](crate::Simulation::apply_command).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn drain_ready<F: FnMut(C)>(&mut self, current_tick: u64, mut apply: F) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].apply_at_tick <= current_tick {
                let cmd = self.pending.remove(i).command;
                apply(cmd);
            } else {
                i += 1;
            }
        }
    }
}

impl<C> Default for CommandQueue<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestCmd {
        SetSpeed(u32),
        Toggle,
    }

    #[test]
    fn new_queue_is_empty() {
        let q: CommandQueue<TestCmd> = CommandQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.current_tick(), 0);
    }

    #[test]
    fn push_now_stamps_current_tick_and_drains_immediately() {
        let mut q: CommandQueue<TestCmd> = CommandQueue::new();
        q.set_current_tick(7);
        q.push_now(TestCmd::Toggle);
        let mut applied: Vec<TestCmd> = Vec::new();
        q.drain_ready(7, |c| applied.push(c));
        assert_eq!(applied, vec![TestCmd::Toggle]);
        assert!(q.is_empty());
    }

    #[test]
    fn future_scheduled_command_stays_queued_until_its_tick() {
        // Acceptance test from the issue: a command with apply_at_tick = N+5
        // queued at tick N is not applied until tick N+5.
        let mut q: CommandQueue<TestCmd> = CommandQueue::new();
        q.set_current_tick(10);
        q.push(15, TestCmd::SetSpeed(2));

        for tick in 10..15 {
            let mut applied: Vec<TestCmd> = Vec::new();
            q.drain_ready(tick, |c| applied.push(c));
            assert!(
                applied.is_empty(),
                "command applied early at tick {tick}: {applied:?}"
            );
        }
        let mut applied: Vec<TestCmd> = Vec::new();
        q.drain_ready(15, |c| applied.push(c));
        assert_eq!(applied, vec![TestCmd::SetSpeed(2)]);
    }

    #[test]
    fn ready_commands_drain_in_insertion_order_across_ticks() {
        // Two ready, one future. Drain at tick 5: ready ones come out in
        // insertion order; future one stays put.
        let mut q: CommandQueue<TestCmd> = CommandQueue::new();
        q.push(3, TestCmd::SetSpeed(1));
        q.push(5, TestCmd::Toggle);
        q.push(10, TestCmd::SetSpeed(4));

        let mut applied: Vec<TestCmd> = Vec::new();
        q.drain_ready(5, |c| applied.push(c));
        assert_eq!(applied, vec![TestCmd::SetSpeed(1), TestCmd::Toggle]);
        assert_eq!(q.len(), 1);

        // The remaining future command drains at tick 10.
        let mut applied: Vec<TestCmd> = Vec::new();
        q.drain_ready(10, |c| applied.push(c));
        assert_eq!(applied, vec![TestCmd::SetSpeed(4)]);
        assert!(q.is_empty());
    }

    #[test]
    fn push_then_push_now_preserves_insertion_order() {
        // Mixing push (explicit tick) and push_now must still drain in
        // insertion order at the same tick.
        let mut q: CommandQueue<TestCmd> = CommandQueue::new();
        q.set_current_tick(4);
        q.push(4, TestCmd::SetSpeed(1));
        q.push_now(TestCmd::Toggle);
        q.push(4, TestCmd::SetSpeed(2));

        let mut applied: Vec<TestCmd> = Vec::new();
        q.drain_ready(4, |c| applied.push(c));
        assert_eq!(
            applied,
            vec![TestCmd::SetSpeed(1), TestCmd::Toggle, TestCmd::SetSpeed(2),]
        );
    }
}
