//! Camera rigs: input-driven controllers that drive a [`Camera`].
//!
//! `Camera` is a pure projection/view-matrix helper. A *rig* sits on top:
//! it owns input-derived state (mouse drags, held keys, scroll wheel) and
//! writes the resulting transform into a `Camera`. This split keeps
//! controller concerns out of the projection helper and leaves room for
//! Cinemachine-style multi-rig setups later (blend between rigs, cut to
//! a cutscene rig, etc.) — each rig is just a value type the [`View`]
//! holds.
//!
//! Today there's exactly one rig — [`OrbitRig`], the standard RTS-style
//! orbit-around-a-focal-point camera with right-click rotate, WASD pan,
//! and scroll-wheel zoom.
//!
//! ## Wiring
//!
//! ```ignore
//! struct MyView {
//!     camera: Camera,
//!     camera_binding: CameraBinding,
//!     rig: OrbitRig,
//! }
//!
//! impl View for MyView {
//!     fn input(&mut self, _: &mut Sim, _: &mut EngineCtx, event: &WindowEvent) {
//!         self.rig.handle_event(event);
//!     }
//!     fn update(&mut self, _: &Sim, _: &mut EngineCtx, dt: Duration) {
//!         self.rig.update(dt);
//!         self.rig.apply_to(&mut self.camera);
//!     }
//!     // ... render: write camera uniform from self.camera, draw.
//! }
//! ```
//!
//! The rig consumes [`WindowEvent`]s in `handle_event` (button state,
//! cursor moves, scroll, WASD key state) and integrates held-key pan in
//! `update` with a wall-clock dt. `apply_to` writes `position`, `target`,
//! and `up` into a [`Camera`].

use std::time::Duration;

use glam::{Vec2, Vec3};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use super::camera::Camera;

/// Tunable parameters for [`OrbitRig`]. Defaults target a roughly
/// table-top-scale strategy view (focus on ground, camera 5–80 units
/// out) and can be overridden field-by-field with struct-update syntax.
#[derive(Clone, Copy, Debug)]
pub struct OrbitRigConfig {
    /// Lower clamp on the camera's elevation angle above the focal
    /// plane, in radians. Keep above zero so the camera never sinks
    /// into the ground.
    pub pitch_min: f32,
    /// Upper clamp on the elevation angle. Keep below `π/2` so the
    /// view never goes degenerate-vertical.
    pub pitch_max: f32,
    /// Lower clamp on the orbit distance.
    pub distance_min: f32,
    /// Upper clamp on the orbit distance.
    pub distance_max: f32,
    /// Radians of rotation per pixel of cursor motion during a
    /// right-button drag.
    pub rotate_sensitivity: f32,
    /// Pan speed (world units per second) at [`pan_distance_ref`]; the
    /// effective speed scales linearly with the current orbit
    /// distance so the felt pan rate stays roughly constant across
    /// zoom levels (Cities Skylines feel).
    pub pan_speed_base: f32,
    /// Reference distance at which [`pan_speed_base`] is the actual
    /// pan speed.
    pub pan_distance_ref: f32,
    /// Multiplicative zoom factor applied per scroll-wheel line.
    /// Scrolling up shrinks the distance by this factor; scrolling
    /// down grows it.
    pub zoom_step: f32,
    /// Invert the vertical drag direction. With `false` (default),
    /// dragging the mouse down raises the camera's elevation — the
    /// view tilts toward top-down, matching the "drag away from
    /// horizon = look more from above" convention. With `true`,
    /// dragging down lowers the camera toward the horizon.
    pub invert_pitch: bool,
    /// Invert the horizontal drag direction. With `false` (default),
    /// dragging the mouse right orbits the camera one way around the
    /// focal point; with `true` it orbits the other way. Pure taste —
    /// flip if the default feels backwards.
    pub invert_yaw: bool,
}

impl OrbitRigConfig {
    pub const DEFAULT: Self = Self {
        pitch_min: 5.0 * std::f32::consts::PI / 180.0,
        pitch_max: 80.0 * std::f32::consts::PI / 180.0,
        distance_min: 2.0,
        distance_max: 80.0,
        rotate_sensitivity: 0.005,
        pan_speed_base: 6.0,
        pan_distance_ref: 10.0,
        zoom_step: 1.15,
        invert_pitch: false,
        invert_yaw: true,
    };
}

impl Default for OrbitRigConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct PanKeys {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
}

/// Strategy-game-style orbit camera rig: focal point on the ground,
/// right-click drag rotates around it, WASD pans the focus, scroll
/// wheel zooms.
///
/// State is conceptually spherical:
/// - `focus` — the point in world space the camera looks at.
/// - `yaw` — rotation around world Z, in radians. `yaw = 0` puts the
///   camera along the +X side of the focal point.
/// - `pitch` — elevation angle of the camera *above* the focal plane,
///   in radians. `pitch = π/2` is directly overhead (clamped away
///   from); `pitch = 0` is on the horizon (also clamped).
/// - `distance` — orbit radius.
///
/// Camera position is derived: `focus + distance * (cos(pitch)cos(yaw),
/// cos(pitch)sin(yaw), sin(pitch))`. The rig is renderer-agnostic — it
/// just writes into a [`Camera`] via [`apply_to`](Self::apply_to).
#[derive(Clone, Debug)]
pub struct OrbitRig {
    pub focus: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub config: OrbitRigConfig,
    rotating: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    keys: PanKeys,
}

impl OrbitRig {
    /// Build a rig focused at `focus` with default tuning. Initial
    /// yaw faces along -X (i.e. camera placed in +X looking back at
    /// the focus), pitch ≈ 45°, distance 12.
    pub fn new(focus: Vec3) -> Self {
        Self {
            focus,
            yaw: 0.0,
            pitch: 45.0_f32.to_radians(),
            distance: 12.0,
            config: OrbitRigConfig::DEFAULT,
            rotating: false,
            last_cursor: None,
            keys: PanKeys::default(),
        }
    }

    /// Ingest a winit window event. Tracks right-mouse-button state,
    /// applies cursor deltas during drag (rotate), accumulates WASD
    /// state, and zooms on scroll. Events the rig doesn't care about
    /// are ignored.
    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                self.rotating = *state == ElementState::Pressed;
                // Reset delta tracking on both press and release: on
                // press we don't have a "previous" cursor yet (would
                // produce a phantom jump from the last release point);
                // on release we just stop tracking.
                self.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } if self.rotating => {
                if let Some(last) = self.last_cursor {
                    let dx = (position.x - last.x) as f32;
                    let dy = (position.y - last.y) as f32;
                    let yaw_sign = if self.config.invert_yaw { -1.0 } else { 1.0 };
                    let pitch_sign = if self.config.invert_pitch { -1.0 } else { 1.0 };
                    self.yaw += yaw_sign * dx * self.config.rotate_sensitivity;
                    self.pitch = (self.pitch + pitch_sign * dy * self.config.rotate_sensitivity)
                        .clamp(self.config.pitch_min, self.config.pitch_max);
                }
                self.last_cursor = Some(*position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // LineDelta is the natural unit; PixelDelta arrives on
                // trackpads — divide by a rough "pixels per line"
                // figure so trackpad scroll has a comparable rate to
                // wheel scroll.
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
                };
                if lines != 0.0 {
                    let factor = self.config.zoom_step.powf(-lines);
                    self.distance = (self.distance * factor)
                        .clamp(self.config.distance_min, self.config.distance_max);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    /// Integrate held-key pan over `dt`. Call once per frame from
    /// [`View::update`](crate::View::update) with the wall-clock dt.
    /// Has no effect when no WASD keys are held.
    pub fn update(&mut self, dt: Duration) {
        let mut input = Vec2::ZERO;
        if self.keys.w {
            input.y += 1.0;
        }
        if self.keys.s {
            input.y -= 1.0;
        }
        if self.keys.d {
            input.x += 1.0;
        }
        if self.keys.a {
            input.x -= 1.0;
        }
        if input == Vec2::ZERO {
            return;
        }
        let input = input.normalize();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        // Camera offset from focus has horizontal direction
        // (cos(yaw), sin(yaw)); look direction (focus minus camera) is
        // the negative of that. Forward = look direction projected to
        // the ground.
        let forward = Vec2::new(-cos_yaw, -sin_yaw);
        // Right = forward rotated -90° around +Z (i.e. (a, b) → (b, -a)).
        let right = Vec2::new(-sin_yaw, cos_yaw);
        let world = forward * input.y + right * input.x;
        let speed = self.config.pan_speed_base * (self.distance / self.config.pan_distance_ref);
        let step = world * speed * dt.as_secs_f32();
        self.focus.x += step.x;
        self.focus.y += step.y;
    }

    /// Compute the camera's world-space position from the current
    /// orbit state.
    pub fn camera_position(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let offset = Vec3::new(cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch);
        self.focus + offset * self.distance
    }

    /// Write the rig's transform into `camera`: sets `position`,
    /// `target`, and `up`. Leaves projection fields (`fov_y_radians`,
    /// `aspect`, `near`, `far`, `zone`) untouched.
    pub fn apply_to(&self, camera: &mut Camera) {
        camera.position = self.camera_position();
        camera.target = self.focus;
        camera.up = Vec3::Z;
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use winit::event::DeviceId;

    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn default_position_is_above_and_offset() {
        let rig = OrbitRig::new(Vec3::ZERO);
        let pos = rig.camera_position();
        // yaw = 0 → camera in +X half; pitch = 45° → x and z share the
        // horizontal radius; y component is zero.
        assert!(pos.x > 0.0);
        assert!(approx_eq(pos.y, 0.0, 1e-5));
        assert!(pos.z > 0.0);
        // Distance to focus matches the configured radius.
        let r = pos.length();
        assert!(approx_eq(r, rig.distance, 1e-4));
    }

    #[test]
    fn pitch_clamp_holds() {
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.pitch = -10.0; // raw write, then handle_event with a big delta
        // Simulate a giant drag-down: handle_event clamps on assignment.
        rig.rotating = true;
        rig.last_cursor = Some(PhysicalPosition::new(0.0, 0.0));
        rig.handle_event(&WindowEvent::CursorMoved {
            device_id: DeviceId::dummy(),
            position: PhysicalPosition::new(0.0, 100_000.0),
        });
        assert!(rig.pitch <= rig.config.pitch_max);
        assert!(rig.pitch >= rig.config.pitch_min - 1e-6);
    }

    #[test]
    fn wasd_pans_in_camera_relative_directions() {
        // At yaw = 0 the camera sits in +X looking at the focus. Forward
        // (pressing W) should move the focus in -X; right (pressing D)
        // should move it in +Y.
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.keys.w = true;
        rig.update(Duration::from_secs_f32(1.0));
        assert!(rig.focus.x < 0.0);
        assert!(approx_eq(rig.focus.y, 0.0, 1e-4));

        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.keys.d = true;
        rig.update(Duration::from_secs_f32(1.0));
        assert!(approx_eq(rig.focus.x, 0.0, 1e-4));
        assert!(rig.focus.y > 0.0);
    }

    #[test]
    fn pan_speed_scales_with_distance() {
        let mut near = OrbitRig::new(Vec3::ZERO);
        let mut far = OrbitRig::new(Vec3::ZERO);
        far.distance = near.distance * 4.0;
        near.keys.w = true;
        far.keys.w = true;
        let dt = Duration::from_secs_f32(1.0);
        near.update(dt);
        far.update(dt);
        // Same time, same direction; far rig should move ~4× as much.
        let near_dx = near.focus.x.abs();
        let far_dx = far.focus.x.abs();
        assert!(far_dx > near_dx * 3.5);
        assert!(far_dx < near_dx * 4.5);
    }

    #[test]
    fn zoom_clamps_and_inverts_correctly() {
        let mut rig = OrbitRig::new(Vec3::ZERO);
        rig.distance = 10.0;
        // Scroll up (positive y) should zoom *in* — reduce distance.
        rig.handle_event(&WindowEvent::MouseWheel {
            device_id: DeviceId::dummy(),
            delta: MouseScrollDelta::LineDelta(0.0, 1.0),
            phase: winit::event::TouchPhase::Started,
        });
        assert!(rig.distance < 10.0);
        // Drive way past the lower clamp.
        for _ in 0..1000 {
            rig.handle_event(&WindowEvent::MouseWheel {
                device_id: DeviceId::dummy(),
                delta: MouseScrollDelta::LineDelta(0.0, 1.0),
                phase: winit::event::TouchPhase::Started,
            });
        }
        assert!(approx_eq(rig.distance, rig.config.distance_min, 1e-4));
    }

    #[test]
    fn invert_pitch_reverses_drag_direction() {
        // Same drag (downward — dy = +100). Default config tilts toward
        // top-down (pitch increases); inverted config tilts toward the
        // horizon (pitch decreases). Start mid-range so neither variant
        // hits a clamp.
        let make_rig = |invert: bool| {
            let mut rig = OrbitRig::new(Vec3::ZERO);
            rig.config.invert_pitch = invert;
            rig.pitch = 45.0_f32.to_radians();
            rig.rotating = true;
            rig.last_cursor = Some(PhysicalPosition::new(0.0, 0.0));
            rig
        };
        let drag_down = WindowEvent::CursorMoved {
            device_id: DeviceId::dummy(),
            position: PhysicalPosition::new(0.0, 100.0),
        };

        let mut default_rig = make_rig(false);
        let p0 = default_rig.pitch;
        default_rig.handle_event(&drag_down);
        assert!(default_rig.pitch > p0);

        let mut inverted = make_rig(true);
        let p0 = inverted.pitch;
        inverted.handle_event(&drag_down);
        assert!(inverted.pitch < p0);
    }

    #[test]
    fn invert_yaw_reverses_drag_direction() {
        // Same drag (rightward — dx = +100). Default config rotates one
        // way; inverted rotates the other. Yaw has no clamp so we just
        // compare signs of the delta.
        let make_rig = |invert: bool| {
            let mut rig = OrbitRig::new(Vec3::ZERO);
            rig.config.invert_yaw = invert;
            rig.rotating = true;
            rig.last_cursor = Some(PhysicalPosition::new(0.0, 0.0));
            rig
        };
        let drag_right = WindowEvent::CursorMoved {
            device_id: DeviceId::dummy(),
            position: PhysicalPosition::new(100.0, 0.0),
        };

        let mut default_rig = make_rig(false);
        let y0 = default_rig.yaw;
        default_rig.handle_event(&drag_right);
        let default_delta = default_rig.yaw - y0;
        assert!(default_delta.abs() > 1e-4);

        let mut inverted = make_rig(true);
        let y0 = inverted.yaw;
        inverted.handle_event(&drag_right);
        let inverted_delta = inverted.yaw - y0;

        // Opposite signs, same magnitude.
        assert!((default_delta + inverted_delta).abs() < 1e-5);
    }

    #[test]
    fn pitch_at_90_camera_is_overhead() {
        let mut rig = OrbitRig::new(Vec3::new(3.0, -2.0, 0.0));
        rig.pitch = FRAC_PI_2 - 1e-4; // just shy of vertical
        let pos = rig.camera_position();
        // Camera sits ~directly over the focus.
        assert!(approx_eq(pos.x, rig.focus.x, 1e-2));
        assert!(approx_eq(pos.y, rig.focus.y, 1e-2));
        assert!(pos.z > rig.focus.z + rig.distance * 0.99);
    }

    #[test]
    fn apply_to_writes_camera_fields() {
        let rig = OrbitRig::new(Vec3::new(5.0, 5.0, 0.0));
        // Marker on a projection field — apply_to must leave it alone.
        let mut cam = Camera {
            fov_y_radians: 1.0,
            ..Camera::default()
        };
        rig.apply_to(&mut cam);
        assert_eq!(cam.target, rig.focus);
        assert_eq!(cam.up, Vec3::Z);
        assert_eq!(cam.position, rig.camera_position());
        // Projection fields untouched.
        assert!(approx_eq(cam.fov_y_radians, 1.0, 1e-6));
    }
}
