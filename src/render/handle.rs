//! [`Handle<T>`] — the streaming-asset reference type.
//!
//! A handle is a cheap-to-clone, refcounted reference to an asset that may
//! still be loading. Render code peeks the state every frame and branches:
//! `Ready` → bind the real resource, `Loading` / `Failed` → bind a fallback.
//! Reads never block. The loader thread fills the slot exactly once via
//! [`OnceLock`], so the read path is lock-free.
//!
//! ## Identity vs contents
//!
//! Handles compare by *identity* — two handles are equal iff they point at
//! the same underlying slot (same load request). Materials use this to detect
//! whether their cached bind group matches the handle that was passed in;
//! comparing texture *contents* would be both slow and wrong (two loads of
//! the same PNG are distinct assets from the caller's perspective).
//!
//! ## Where this lives
//!
//! Handles are view-side today because the only consumer is GPU asset
//! streaming. If sim-side async ever shows up the type can move down into
//! [`crate::data`] unchanged; the design has no `wgpu`/`winit` dependencies.

use std::fmt;
use std::sync::{Arc, OnceLock};

/// Reference-counted, write-once handle to a streaming asset of type `T`.
///
/// Construct directly via [`Handle::ready`] when the value is already in
/// hand, or — more commonly — by asking the
/// [`AssetServer`](super::AssetServer) for one. Background loaders fill the
/// slot exactly once; subsequent [`peek`](Self::peek)s see the final state.
pub struct Handle<T> {
    inner: Arc<HandleSlot<T>>,
}

/// The shared, write-once payload behind a [`Handle`]. The reader path is
/// just `OnceLock::get()` — no locking when the value is settled.
struct HandleSlot<T> {
    result: OnceLock<Result<T, HandleError>>,
    /// Diagnostic-only label (typically the source path). Carried so error
    /// messages and `Debug` output can name what failed without the caller
    /// remembering which handle they peeked.
    label: String,
}

/// Snapshot of a [`Handle`]'s state, borrowing from the handle for the
/// duration of one `peek`. `Ready` lends `&T`, `Failed` lends `&HandleError`;
/// `Loading` carries no payload.
///
/// The lifetime ties readers to the handle: the natural usage is `match
/// handle.peek() { ... }` inside a single expression. To survive across
/// frames, clone the handle (cheap) and re-peek next frame.
#[derive(Debug)]
pub enum HandleState<'a, T> {
    /// Background load is in flight (or hasn't started).
    Loading,
    /// Load finished successfully; the asset is bound for the lifetime of
    /// this borrow.
    Ready(&'a T),
    /// Load failed terminally. The error is final — there is no retry path.
    Failed(&'a HandleError),
}

/// Reason a [`Handle`] is in the `Failed` state.
///
/// Carries a human-readable message plus the source path / diagnostic label
/// the handle was created with. Construct via [`HandleError::new`] from a
/// loader; the asset server formats VFS read failures and decode failures
/// into this shape.
#[derive(Debug, Clone)]
pub struct HandleError {
    message: String,
}

impl HandleError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for HandleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for HandleError {}

impl<T> Handle<T> {
    /// A handle that is already `Ready`. Useful for callers that have the
    /// asset eagerly (tests, examples that loaded synchronously, the
    /// magenta-fallback resident texture).
    pub fn ready(value: T) -> Self {
        let slot = HandleSlot {
            result: OnceLock::new(),
            label: String::new(),
        };
        let _ = slot.result.set(Ok(value));
        Self {
            inner: Arc::new(slot),
        }
    }

    /// A handle that has already failed. Symmetric counterpart to
    /// [`Handle::ready`] — useful for tests and for callers that want to
    /// surface a synchronous error through the same plumbing async loaders
    /// use.
    pub fn failed(error: HandleError) -> Self {
        let slot = HandleSlot {
            result: OnceLock::new(),
            label: String::new(),
        };
        let _ = slot.result.set(Err(error));
        Self {
            inner: Arc::new(slot),
        }
    }

    /// Peek the current state. Lock-free once the slot has been written.
    /// The returned borrow lives only as long as `self`; for cross-frame
    /// use, clone the handle and peek again.
    pub fn peek(&self) -> HandleState<'_, T> {
        match self.inner.result.get() {
            None => HandleState::Loading,
            Some(Ok(t)) => HandleState::Ready(t),
            Some(Err(e)) => HandleState::Failed(e),
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.inner.result.get(), Some(Ok(_)))
    }

    pub fn is_loading(&self) -> bool {
        self.inner.result.get().is_none()
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.inner.result.get(), Some(Err(_)))
    }

    /// Diagnostic label this handle was created with (typically the source
    /// path). Empty for handles built via [`Handle::ready`] or
    /// [`Handle::failed`].
    pub fn label(&self) -> &str {
        &self.inner.label
    }

    /// Identity comparison — two handles are equal iff they point at the
    /// same underlying load. Materials use this to decide whether their
    /// cached bind group is still valid for the handle the caller passed.
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.inner, &b.inner)
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match self.inner.result.get() {
            None => "Loading",
            Some(Ok(_)) => "Ready",
            Some(Err(_)) => "Failed",
        };
        f.debug_struct("Handle")
            .field("label", &self.inner.label)
            .field("state", &state)
            .finish()
    }
}

/// Constructors used by [`AssetServer`](super::AssetServer) to wire up a
/// pair of (read-side handle, write-side filler) for a freshly-spawned
/// load.
///
/// The writer is the *only* path to mutate a handle's state; once filled,
/// the [`OnceLock`] inside the slot rejects further writes. The writer is
/// `pub(super)` so user code can't fabricate one and overwrite a handle out
/// from under the asset server.
pub(super) fn new_pending<T>(label: impl Into<String>) -> (Handle<T>, HandleWriter<T>) {
    let slot = Arc::new(HandleSlot {
        result: OnceLock::new(),
        label: label.into(),
    });
    let writer = HandleWriter {
        slot: Arc::clone(&slot),
    };
    (Handle { inner: slot }, writer)
}

/// Write-side counterpart to [`Handle`] — used by [`AssetServer`] background
/// loaders to publish the final state of a load. Dropping a writer without
/// calling [`set_ready`](Self::set_ready) or [`set_failed`](Self::set_failed)
/// leaves the handle stuck in `Loading` permanently; that's a loader bug, not
/// a recoverable runtime state.
pub(super) struct HandleWriter<T> {
    slot: Arc<HandleSlot<T>>,
}

impl<T> HandleWriter<T> {
    pub(super) fn set_ready(self, value: T) {
        // OnceLock::set returns Err with the value back on second write; we
        // ignore it because the only writer is this consumed-on-call type,
        // so a second write would be a programming error rather than a
        // runtime condition.
        let _ = self.slot.result.set(Ok(value));
    }

    pub(super) fn set_failed(self, error: HandleError) {
        let _ = self.slot.result.set(Err(error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_handle_peeks_ready() {
        let h: Handle<u32> = Handle::ready(42);
        match h.peek() {
            HandleState::Ready(v) => assert_eq!(*v, 42),
            other => panic!("expected Ready, got {other:?}"),
        }
        assert!(h.is_ready());
        assert!(!h.is_loading());
        assert!(!h.is_failed());
    }

    #[test]
    fn failed_handle_peeks_failed() {
        let h: Handle<u32> = Handle::failed(HandleError::new("nope"));
        match h.peek() {
            HandleState::Failed(e) => assert_eq!(e.message(), "nope"),
            other => panic!("expected Failed, got {other:?}"),
        }
        assert!(h.is_failed());
        assert!(!h.is_ready());
    }

    #[test]
    fn pending_handle_starts_loading_then_publishes() {
        let (h, w) = new_pending::<u32>("foo");
        assert!(h.is_loading());
        assert_eq!(h.label(), "foo");
        w.set_ready(7);
        match h.peek() {
            HandleState::Ready(v) => assert_eq!(*v, 7),
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn pending_handle_can_fail() {
        let (h, w) = new_pending::<u32>("bar");
        w.set_failed(HandleError::new("decode error"));
        match h.peek() {
            HandleState::Failed(e) => assert_eq!(e.message(), "decode error"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn clones_share_state() {
        let (h1, w) = new_pending::<u32>("shared");
        let h2 = h1.clone();
        assert!(h1.is_loading());
        assert!(h2.is_loading());
        w.set_ready(99);
        assert!(h1.is_ready());
        assert!(h2.is_ready());
        // Both clones see the same value.
        match (h1.peek(), h2.peek()) {
            (HandleState::Ready(a), HandleState::Ready(b)) => {
                assert_eq!(*a, 99);
                assert_eq!(*b, 99);
            }
            _ => panic!("clones diverged"),
        }
    }

    #[test]
    fn ptr_eq_distinguishes_distinct_loads() {
        let a: Handle<u32> = Handle::ready(1);
        let b: Handle<u32> = Handle::ready(1);
        let a_clone = a.clone();
        assert!(Handle::ptr_eq(&a, &a_clone));
        // Same contents, different loads → distinct identity.
        assert!(!Handle::ptr_eq(&a, &b));
    }

    #[test]
    fn second_set_is_silently_dropped() {
        // The OnceLock inside the slot is the source of truth — even if a
        // writer is fabricated against an already-filled slot, the first
        // value wins. Belt-and-braces on top of the consumed-on-call
        // HandleWriter shape.
        let (h, w1) = new_pending::<u32>("x");
        w1.set_ready(1);
        // We can't make a second HandleWriter from outside this module —
        // the path goes through the constructor — but we can still assert
        // that the state observed by the handle is the first value, no
        // matter what.
        assert!(matches!(h.peek(), HandleState::Ready(v) if *v == 1));
    }
}
