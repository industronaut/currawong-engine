//! Yaw-only quantized angle for sim-side rotation.
//!
//! Sim objects in this engine never need free 3D rotation — pawns face one
//! of 8 or 16 directions, buildings snap to 4 orientations, trees have no
//! sim rotation at all. Free [`Quat`](glam::Quat) rotation is a view need
//! (smooth interpolation between sim states), not a sim need. [`Facing`] is
//! the constrained shape: 16-bit integer angle that wraps automatically,
//! bit-exact across architectures, no `f32` involved.
//!
//! The full circle is `2^16 = 65536` units, so 1 unit ≈ 0.0055°. Equality
//! and addition are wrapping integer ops on the underlying `u16`. Cardinal
//! quantization (`to_cardinal8` / `to_cardinal16`) is a single right-shift.
//!
//! ## Atan2
//!
//! [`Facing::from_direction`] converts a [`SimVec`] into a quantized angle
//! via a fixed-point integer atan2 (octant reduction + polynomial). This
//! is the "lookup table when sim trig is needed" the determinism issue
//! flagged — `lumber_camp`'s motion module wants it today for velocity →
//! facing, so it lands now. Pure integer math, deterministic across
//! platforms.

use core::ops::{Add, AddAssign, Sub, SubAssign};

use glam::Quat;

use super::units::{SimUnit, SimVec};

/// Quantized yaw-only angle. One full revolution is `Facing::FULL_TURN`
/// (= `u16::MAX + 1`), so addition is wrapping `u16` arithmetic and
/// equality is integer equality.
///
/// Zero points along the +X axis; positive direction rotates toward +Y
/// (counterclockwise viewed from +Z, matching the right-handed Z-up frame).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Facing(pub u16);

impl Facing {
    /// Number of integer units in a full revolution. The full circle wraps
    /// at `u16::MAX + 1 = 65536`; values outside `[0, 2^16)` aren't
    /// representable. `FULL_TURN as u16` overflows, so this is `u32`.
    pub const FULL_TURN: u32 = 1 << 16;

    pub const ZERO: Self = Facing(0);
    /// 90° — +X to +Y.
    pub const QUARTER: Self = Facing(1 << 14);
    /// 180°.
    pub const HALF: Self = Facing(1 << 15);
    /// 270°.
    pub const THREE_QUARTERS: Self = Facing((1 << 14) * 3);

    /// Quantize this facing into one of 8 cardinal directions: 0..=7,
    /// counterclockwise from +X (E, NE, N, NW, W, SW, S, SE in the
    /// engine's X-right Y-forward Z-up frame).
    #[inline]
    pub const fn to_cardinal8(self) -> u8 {
        // Add half a sector before truncating so each sector is centred on
        // its cardinal direction (e.g. due-east accepts ±22.5°).
        let half_sector = 1 << 12; // 2^16 / 16
        ((self.0.wrapping_add(half_sector)) >> 13) as u8 & 0b111
    }

    /// Quantize into 16 directions (22.5° each). 0..=15, same orientation
    /// as [`Self::to_cardinal8`].
    #[inline]
    pub const fn to_cardinal16(self) -> u8 {
        let half_sector = 1 << 11;
        ((self.0.wrapping_add(half_sector)) >> 12) as u8 & 0b1111
    }

    /// Angle in radians. The view seam — sim code never needs this. The
    /// `f32` here is the unavoidable cost of building a [`Quat`] for the
    /// renderer; the determinism boundary is one-way (sim is integer,
    /// view accepts the float conversion at extract time).
    #[inline]
    pub fn to_radians(self) -> f32 {
        // Map [0, 65536) → [0, TAU) exactly via 1/65536 * TAU.
        (self.0 as f32) * (core::f32::consts::TAU / Self::FULL_TURN as f32)
    }

    /// Build a Z-axis [`Quat`] at this facing — the standard view-side
    /// conversion. Use at extract time when building a model matrix.
    #[inline]
    pub fn to_quat(self) -> Quat {
        Quat::from_rotation_z(self.to_radians())
    }

    /// Build from a radian angle. For test fixtures / one-off conversion
    /// from authored float data; production sim code constructs from
    /// [`Self::from_direction`] or [`Self::ZERO`].
    #[inline]
    pub fn from_radians(theta: f32) -> Self {
        let scale = Self::FULL_TURN as f32 / core::f32::consts::TAU;
        // rem_euclid keeps the result in [0, FULL_TURN); the `as u32` cast
        // truncates toward zero (already non-negative after rem_euclid).
        let raw = (theta * scale).rem_euclid(Self::FULL_TURN as f32) as u32;
        Facing(raw as u16)
    }

    /// Build from a direction expressed as a [`SimVec`]. Uses an integer
    /// atan2 on the (x, y) plane (z is ignored — yaw-only). Bit-exact
    /// across architectures.
    ///
    /// Returns [`Facing::ZERO`] for the degenerate zero-length direction.
    #[inline]
    pub fn from_direction(dir: SimVec) -> Self {
        Self::atan2(dir.y, dir.x)
    }

    /// Integer atan2 producing a [`Facing`]. Octant reduction + a
    /// degree-3 polynomial fit of `atan(z)` over `[0, 1]` to Facing
    /// units. Bit-exact, no trig calls, no float ops. Max error over the
    /// full circle is ~0.1° (~18 Facing units) — well below the 0.5° /
    /// 90 Facing units that the cardinal-quantizer cares about.
    #[inline]
    pub fn atan2(y: SimUnit, x: SimUnit) -> Self {
        let xb = x.to_bits();
        let yb = y.to_bits();
        if xb == 0 && yb == 0 {
            return Facing::ZERO;
        }
        // Work in i64 so the polynomial's products don't overflow. `abs()`
        // of i32::MIN would wrap; widen first.
        let ax = (xb as i64).abs();
        let ay = (yb as i64).abs();
        // Octant reduction: compute atan(z) where z = min/max ∈ [0, 1].
        // If we swapped (ay > ax), the true angle lives in [45°, 90°] and
        // we reflect later.
        let (small, large) = if ay <= ax { (ay, ax) } else { (ax, ay) };
        // z in Q16.16: 65536 represents 1.0.
        let z_q16: i64 = (small << 16) / large;

        // Polynomial fit on [0, 1]:
        //   atan(z) ≈ z * (π/4 + (1 - z) * (0.2447 + 0.0663 * z))
        // This is the standard minimax form (max abs error ~0.0015 rad).
        // Scale every constant by 2^16 / TAU = 10430.378 so the result
        // comes out directly in Facing units:
        //   π/4 → 8192 (1/8 turn)
        //   0.2447 → 2552
        //   0.0663 → 691
        const PI_QUARTER_FU: i64 = 8192;
        const A_FU: i64 = 2552;
        const B_FU: i64 = 691;
        // inner = A + B*z (in Facing units, after shifting out the Q16 of z).
        let inner = A_FU + ((B_FU * z_q16) >> 16);
        let one_minus_z = (1_i64 << 16) - z_q16; // Q16.16
        // correction = (1 - z) * inner (Facing units after shifting Q16 of (1-z)).
        let correction = (one_minus_z * inner) >> 16;
        let bracket = PI_QUARTER_FU + correction;
        // result = z * bracket (Facing units after shifting Q16 of z).
        let mut a: i64 = (z_q16 * bracket) >> 16;

        // Reflect [0°, 45°] → [45°, 90°] if we swapped.
        if ay > ax {
            a = 16384 - a;
        }
        // Quadrant reflection by sign. 32768 = 180°, 65536 = 360° = 0°.
        let result: i64 = match (xb >= 0, yb >= 0) {
            (true, true) => a,           // Q1: +x, +y → as-is
            (false, true) => 32768 - a,  // Q2: -x, +y → 180° - a
            (false, false) => 32768 + a, // Q3: -x, -y → 180° + a
            (true, false) => 65536 - a,  // Q4: +x, -y → 360° - a
        };
        Facing((result & 0xFFFF) as u16)
    }
}

// --- Arithmetic ---------------------------------------------------------
//
// Wrapping `u16` ops. Adding two facings gives a facing (rotation
// composition); subtracting gives the signed delta as a facing (so 350° -
// 10° wraps cleanly to 340° rather than going negative).

impl Add for Facing {
    type Output = Facing;
    #[inline]
    fn add(self, rhs: Facing) -> Facing {
        Facing(self.0.wrapping_add(rhs.0))
    }
}

impl AddAssign for Facing {
    #[inline]
    fn add_assign(&mut self, rhs: Facing) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl Sub for Facing {
    type Output = Facing;
    #[inline]
    fn sub(self, rhs: Facing) -> Facing {
        Facing(self.0.wrapping_sub(rhs.0))
    }
}

impl SubAssign for Facing {
    #[inline]
    fn sub_assign(&mut self, rhs: Facing) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(d: f32) -> Facing {
        Facing::from_radians(d.to_radians())
    }

    fn close_units(a: Facing, b: Facing, slack: i32) -> bool {
        // Compare with wrap-aware delta. `b - a` gives shortest positive
        // wrapping distance; reinterpret as signed.
        let raw = (b - a).0 as i32;
        let signed = if raw > 32_768 { raw - 65_536 } else { raw };
        signed.abs() <= slack
    }

    #[test]
    fn wrap_around_is_zero() {
        let a = Facing(60_000);
        let b = Facing(10_000);
        // 60000 + 10000 = 70000 % 65536 = 4464
        assert_eq!((a + b).0, 4464);
    }

    #[test]
    fn cardinal8_quantizes_to_nearest() {
        assert_eq!(Facing::ZERO.to_cardinal8(), 0); // +X
        assert_eq!(Facing::QUARTER.to_cardinal8(), 2); // +Y
        assert_eq!(Facing::HALF.to_cardinal8(), 4);
        assert_eq!(Facing::THREE_QUARTERS.to_cardinal8(), 6);
        // Half a sector past +X should still be 0.
        assert_eq!(Facing(4095).to_cardinal8(), 0);
        // Beyond the half-sector boundary it ticks to 1 (NE).
        assert_eq!(Facing(8192).to_cardinal8(), 1);
    }

    #[test]
    fn cardinal16_is_finer_grid() {
        assert_eq!(Facing::ZERO.to_cardinal16(), 0);
        assert_eq!(Facing::QUARTER.to_cardinal16(), 4);
        assert_eq!(Facing::HALF.to_cardinal16(), 8);
        assert_eq!(Facing::THREE_QUARTERS.to_cardinal16(), 12);
    }

    #[test]
    fn from_radians_round_trips_cardinals() {
        // Exact: 90° → QUARTER.
        assert_eq!(Facing::from_radians(0.0), Facing::ZERO);
        // Allow a couple of Q16 units of slack — float quantization.
        assert!(close_units(deg(90.0), Facing::QUARTER, 2));
        assert!(close_units(deg(180.0), Facing::HALF, 2));
        assert!(close_units(deg(270.0), Facing::THREE_QUARTERS, 2));
    }

    #[test]
    fn atan2_cardinals_exact() {
        let unit = SimUnit::from_num(1);
        let zero = SimUnit::ZERO;
        assert!(close_units(Facing::atan2(zero, unit), Facing::ZERO, 4));
        assert!(close_units(Facing::atan2(unit, zero), Facing::QUARTER, 4));
        assert!(close_units(Facing::atan2(zero, -unit), Facing::HALF, 4));
        assert!(close_units(
            Facing::atan2(-unit, zero),
            Facing::THREE_QUARTERS,
            4
        ));
    }

    #[test]
    fn atan2_45_degrees() {
        let unit = SimUnit::from_num(1);
        // 45° = QUARTER / 2 = 8192.
        let target = Facing(8192);
        assert!(
            close_units(Facing::atan2(unit, unit), target, 30),
            "atan2(1,1) = {:?}",
            Facing::atan2(unit, unit)
        );
    }

    #[test]
    fn atan2_matches_floating_point_within_slack() {
        // Sweep angles, compare to f32::atan2. The integer version is allowed
        // to differ by up to ~50 Facing units (~0.27°) — well inside the
        // visual quantization budget.
        for d in (-180..180).step_by(7) {
            let theta = (d as f32).to_radians();
            let y = SimUnit::from_num(theta.sin());
            let x = SimUnit::from_num(theta.cos());
            let got = Facing::atan2(y, x);
            let expected = Facing::from_radians(theta);
            assert!(
                close_units(got, expected, 60),
                "atan2 at {d}°: got {got:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn atan2_zero_input_is_zero() {
        assert_eq!(Facing::atan2(SimUnit::ZERO, SimUnit::ZERO), Facing::ZERO);
    }

    #[test]
    fn atan2_is_deterministic() {
        // The whole point. Same inputs → same outputs, repeated.
        let unit = SimUnit::from_num(1);
        let a = Facing::atan2(unit + SimUnit::from_num(0.3), unit - SimUnit::from_num(0.7));
        let b = Facing::atan2(unit + SimUnit::from_num(0.3), unit - SimUnit::from_num(0.7));
        assert_eq!(a, b);
    }

    #[test]
    fn from_direction_uses_xy_plane_only() {
        // Z is ignored.
        let pure_x = Facing::from_direction(SimVec::tile(1, 0, 0));
        let with_z = Facing::from_direction(SimVec::tile(1, 0, 99));
        assert_eq!(pure_x, with_z);
    }

    #[test]
    fn arithmetic_wraps_cleanly() {
        let a = Facing::HALF + Facing::HALF;
        assert_eq!(a, Facing::ZERO);
        let b = Facing::ZERO - Facing::QUARTER;
        assert_eq!(b, Facing::THREE_QUARTERS);
    }
}
