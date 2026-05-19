//! Fixed-point integer types for sim-side position and offset state.
//!
//! Two design forces converge on integers: (1) genre fit — DF / Factorio /
//! RimWorld are quantized worlds where tile-integer + small sub-tile
//! fractional is the natural shape; (2) determinism — integer arithmetic is
//! bit-exact across CPUs, compilers, and optimization levels, where `f32`
//! transcendentals can diverge in the last bit and FMA contraction can
//! reorder operations.
//!
//! Prior art for folding tile and sub-tile into one fixed value: OpenRA's
//! `WPos` (i32, 1024 = 1 tile), Factorio's `Position` (i32, 256 = 1 tile),
//! TrueSync / Photon Quantum throughout.
//!
//! ## Shape
//!
//! - [`SimUnit`] — base scalar, Q16.16 (`FixedI32<U16>`). ±32,768 tile range,
//!   1/65,536 tile resolution. `SimUnit * SimUnit` widens through `i64`
//!   internally before truncating; the `fixed` crate handles that, so the
//!   1-tile-precision over-spend buys multiplication headroom.
//! - [`SimPos`] — position in a zone-local frame. Three [`SimUnit`]s.
//! - [`SimVec`] — relative offset. Same internal shape as [`SimPos`] but a
//!   separate type so the type system rejects `pos + pos`. `pos + vec → pos`,
//!   `pos - pos → vec`, `vec * scalar → vec`.
//!
//! ## Conversion to view
//!
//! The sim/view seam stays at the view's extract step. Use [`SimPos::to_vec3`]
//! / [`SimVec::to_vec3`] there. The reverse [`SimPos::from_vec3`] exists for
//! tests / one-off sim construction from authored data; production sim code
//! builds positions from integers ([`SimPos::tile`], [`SimPos::splat_tile`],
//! [`From<(i32, i32, i32)>`]).

use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use fixed::FixedI32;
use fixed::types::extra::U16;
use glam::Vec3;

/// Scalar sim-side unit: signed Q16.16 fixed-point. One unit equals one
/// tile; the fractional 16 bits address sub-tile position to 1/65,536 of
/// a tile.
///
/// Bit-exact across architectures: every arithmetic operation is integer.
/// Multiplication widens through `i64` internally and saturates on overflow
/// — the `fixed` crate's default policy. The ±32,768 tile range is the
/// architecturally meaningful number; a sim that needs more should not be
/// using one zone.
pub type SimUnit = FixedI32<U16>;

/// Construct a [`SimUnit`] from an integer tile count. Shorthand for
/// `SimUnit::from_num(n)` — the common case is "exactly N tiles from
/// origin", so call sites read better as `tile(3)` than as a from_num.
#[inline]
pub const fn tile(n: i32) -> SimUnit {
    SimUnit::const_from_int(n)
}

/// Position in a zone's local frame. Three [`SimUnit`]s; the engine is
/// right-handed Z-up (X/Y are the ground plane, Z is height).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimPos {
    pub x: SimUnit,
    pub y: SimUnit,
    pub z: SimUnit,
}

impl SimPos {
    /// The zone origin. All-zero coordinates.
    pub const ZERO: Self = Self {
        x: SimUnit::ZERO,
        y: SimUnit::ZERO,
        z: SimUnit::ZERO,
    };

    /// Build from raw [`SimUnit`] components. The general-purpose constructor.
    #[inline]
    pub const fn new(x: SimUnit, y: SimUnit, z: SimUnit) -> Self {
        Self { x, y, z }
    }

    /// Build from integer tile coordinates — the common case. Equivalent to
    /// `SimPos::new(tile(x), tile(y), tile(z))`.
    #[inline]
    pub const fn tile(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: tile(x),
            y: tile(y),
            z: tile(z),
        }
    }

    /// Build a position at the centre of the tile `(x, y)` at height z. The
    /// half-tile offset is the convention most examples want; explicit so
    /// it doesn't get baked into [`Self::tile`] (which is exactly on-grid).
    #[inline]
    pub fn tile_center(x: i32, y: i32, z: i32) -> Self {
        let half = SimUnit::from_num(0.5);
        Self {
            x: tile(x) + half,
            y: tile(y) + half,
            z: tile(z),
        }
    }

    /// Build a position with the same value on all three axes.
    #[inline]
    pub const fn splat_tile(n: i32) -> Self {
        Self::tile(n, n, n)
    }

    /// Decompose into integer tile coordinates by truncating each axis
    /// toward negative infinity (Euclidean for symmetry across the origin).
    /// `pos.tile()` for `SimPos { x: tile(-1) + 0.25, ... }` is `(-1, ..)`.
    #[inline]
    pub fn tile_coord(self) -> (i32, i32, i32) {
        (
            self.x.to_num::<i32>(),
            self.y.to_num::<i32>(),
            self.z.to_num::<i32>(),
        )
    }

    /// The sub-tile fractional part of each axis, as a [`SimVec`].
    /// `pos - SimPos::tile(pos.tile_coord())`.
    #[inline]
    pub fn sub_tile(self) -> SimVec {
        let (tx, ty, tz) = self.tile_coord();
        SimVec {
            x: self.x - tile(tx),
            y: self.y - tile(ty),
            z: self.z - tile(tz),
        }
    }

    /// View seam: convert to `glam::Vec3` for the renderer. The sim stays
    /// integer; this is the one-way exit.
    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(
            self.x.to_num::<f32>(),
            self.y.to_num::<f32>(),
            self.z.to_num::<f32>(),
        )
    }

    /// Build from `glam::Vec3`. Saturates to the [`SimUnit`] range if the
    /// input is out of bounds. For test fixtures and one-off conversion from
    /// authored float data only — production sim code constructs from
    /// integer tiles.
    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self {
            x: SimUnit::saturating_from_num(v.x),
            y: SimUnit::saturating_from_num(v.y),
            z: SimUnit::saturating_from_num(v.z),
        }
    }
}

impl From<(i32, i32, i32)> for SimPos {
    #[inline]
    fn from((x, y, z): (i32, i32, i32)) -> Self {
        Self::tile(x, y, z)
    }
}

/// Relative offset in a zone's local frame. Three [`SimUnit`]s. Distinct
/// from [`SimPos`] so the type system rejects nonsensical operations like
/// adding two positions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SimVec {
    pub x: SimUnit,
    pub y: SimUnit,
    pub z: SimUnit,
}

impl SimVec {
    pub const ZERO: Self = Self {
        x: SimUnit::ZERO,
        y: SimUnit::ZERO,
        z: SimUnit::ZERO,
    };

    #[inline]
    pub const fn new(x: SimUnit, y: SimUnit, z: SimUnit) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn tile(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: tile(x),
            y: tile(y),
            z: tile(z),
        }
    }

    /// Squared length. Sum of squared axes, in widened i64 internally;
    /// caller decides what to do with the [`SimUnit`] result.
    #[inline]
    pub fn length_squared(self) -> SimUnit {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Length via integer sqrt of `length_squared`. Bit-exact across
    /// platforms (no `f32::sqrt` involvement). Use this for distance
    /// comparisons; for vector normalisation, prefer keeping the
    /// computation in squared form when you can.
    #[inline]
    pub fn length(self) -> SimUnit {
        let sq = self.length_squared();
        // Q16.16 sqrt: sqrt of a Q16.16 number gives a Q8.8 number, so we
        // shift bits to keep the fractional precision balanced. Equivalent
        // to `((raw as i64) << 16).isqrt() as i32` packaged as SimUnit.
        let raw = sq.to_bits().max(0) as u64;
        let widened = raw << 16;
        let root = integer_sqrt_u64(widened) as i32;
        SimUnit::from_bits(root)
    }

    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(
            self.x.to_num::<f32>(),
            self.y.to_num::<f32>(),
            self.z.to_num::<f32>(),
        )
    }

    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self {
            x: SimUnit::saturating_from_num(v.x),
            y: SimUnit::saturating_from_num(v.y),
            z: SimUnit::saturating_from_num(v.z),
        }
    }
}

/// Newton-style integer sqrt of a u64, bit-exact across platforms. Returns
/// floor(sqrt(n)). Used by [`SimVec::length`].
#[inline]
fn integer_sqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    // Initial guess: 2^(bits/2) gives the right order of magnitude.
    let bits = 64 - n.leading_zeros();
    let mut x = 1u64 << bits.div_ceil(2);
    loop {
        let next = (x + n / x) / 2;
        if next >= x {
            return x;
        }
        x = next;
    }
}

// --- Arithmetic ---------------------------------------------------------

impl Add<SimVec> for SimPos {
    type Output = SimPos;
    #[inline]
    fn add(self, rhs: SimVec) -> SimPos {
        SimPos {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign<SimVec> for SimPos {
    #[inline]
    fn add_assign(&mut self, rhs: SimVec) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub<SimVec> for SimPos {
    type Output = SimPos;
    #[inline]
    fn sub(self, rhs: SimVec) -> SimPos {
        SimPos {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign<SimVec> for SimPos {
    #[inline]
    fn sub_assign(&mut self, rhs: SimVec) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Sub<SimPos> for SimPos {
    type Output = SimVec;
    #[inline]
    fn sub(self, rhs: SimPos) -> SimVec {
        SimVec {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Add<SimVec> for SimVec {
    type Output = SimVec;
    #[inline]
    fn add(self, rhs: SimVec) -> SimVec {
        SimVec {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign<SimVec> for SimVec {
    #[inline]
    fn add_assign(&mut self, rhs: SimVec) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub<SimVec> for SimVec {
    type Output = SimVec;
    #[inline]
    fn sub(self, rhs: SimVec) -> SimVec {
        SimVec {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign<SimVec> for SimVec {
    #[inline]
    fn sub_assign(&mut self, rhs: SimVec) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Neg for SimVec {
    type Output = SimVec;
    #[inline]
    fn neg(self) -> SimVec {
        SimVec {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<SimUnit> for SimVec {
    type Output = SimVec;
    #[inline]
    fn mul(self, rhs: SimUnit) -> SimVec {
        SimVec {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<SimUnit> for SimVec {
    type Output = SimVec;
    #[inline]
    fn div(self, rhs: SimUnit) -> SimVec {
        SimVec {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_constructor_is_exact() {
        let p = SimPos::tile(3, 5, -2);
        assert_eq!(p.x, SimUnit::from_num(3));
        assert_eq!(p.y, SimUnit::from_num(5));
        assert_eq!(p.z, SimUnit::from_num(-2));
    }

    #[test]
    fn tile_coord_truncates_toward_negative_infinity() {
        let p = SimPos {
            x: tile(-1) + SimUnit::from_num(0.25),
            y: tile(2) + SimUnit::from_num(0.75),
            z: SimUnit::ZERO,
        };
        assert_eq!(p.tile_coord(), (-1, 2, 0));
    }

    #[test]
    fn sub_tile_recovers_fractional() {
        let frac = SimUnit::from_num(0.25);
        let p = SimPos {
            x: tile(7) + frac,
            y: tile(0),
            z: tile(0),
        };
        let sub = p.sub_tile();
        assert_eq!(sub.x, frac);
        assert_eq!(sub.y, SimUnit::ZERO);
        assert_eq!(sub.z, SimUnit::ZERO);
    }

    #[test]
    fn pos_minus_pos_is_vec() {
        let a = SimPos::tile(5, 5, 0);
        let b = SimPos::tile(2, 1, 0);
        let v: SimVec = a - b;
        assert_eq!(v, SimVec::tile(3, 4, 0));
    }

    #[test]
    fn pos_plus_vec_round_trips() {
        let p = SimPos::tile(10, 10, 10);
        let v = SimVec::tile(1, 2, 3);
        let q = p + v;
        assert_eq!(q - v, p);
        assert_eq!(q - p, v);
    }

    #[test]
    fn vec_length_squared_is_bit_exact() {
        // 3-4-5 right triangle.
        let v = SimVec::tile(3, 4, 0);
        assert_eq!(v.length_squared(), SimUnit::from_num(25));
    }

    #[test]
    fn vec_length_matches_pythagoras() {
        let v = SimVec::tile(3, 4, 0);
        let l = v.length();
        // Within Q16.16 quantization (1/65536 tile).
        let err = (l - SimUnit::from_num(5)).abs();
        assert!(err <= SimUnit::from_bits(2), "len {l} too far from 5");
    }

    #[test]
    fn vec_length_zero_is_zero() {
        assert_eq!(SimVec::ZERO.length(), SimUnit::ZERO);
        assert_eq!(SimVec::ZERO.length_squared(), SimUnit::ZERO);
    }

    #[test]
    fn vec_arithmetic_is_deterministic_across_repeats() {
        // Same sequence of operations from the same starting state must
        // produce the same final state. This is the determinism core.
        fn step(seed: SimVec) -> SimVec {
            let mut v = seed;
            for i in 0..1000 {
                let s = SimUnit::from_num(i % 7) - SimUnit::from_num(3);
                v += SimVec::tile(1, 2, 3) * s;
                v -= SimVec::new(SimUnit::from_num(0.125), SimUnit::ZERO, SimUnit::ZERO);
            }
            v
        }
        let a = step(SimVec::ZERO);
        let b = step(SimVec::ZERO);
        assert_eq!(a, b);
    }

    #[test]
    fn pos_to_vec3_round_trips_integer_values() {
        let p = SimPos::tile(7, -3, 2);
        let v = p.to_vec3();
        assert_eq!(v, Vec3::new(7.0, -3.0, 2.0));
        assert_eq!(SimPos::from_vec3(v), p);
    }

    #[test]
    fn from_vec3_saturates_out_of_range() {
        // Q16.16 caps at ±32,768.
        let huge = Vec3::new(1.0e10, -1.0e10, 0.0);
        let p = SimPos::from_vec3(huge);
        assert!(p.x > SimUnit::from_num(32_000));
        assert!(p.y < SimUnit::from_num(-32_000));
    }

    #[test]
    fn integer_sqrt_matches_floor_sqrt() {
        for n in [0u64, 1, 2, 3, 4, 99, 100, 101, 1_000_000, u64::MAX >> 2] {
            let s = integer_sqrt_u64(n);
            assert!(s * s <= n, "sqrt({n}) = {s}; {s}^2 > {n}");
            assert!(
                s.saturating_add(1).saturating_mul(s.saturating_add(1)) > n,
                "sqrt({n}) = {s}; (s+1)^2 <= n"
            );
        }
    }
}
