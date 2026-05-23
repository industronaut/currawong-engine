//! [`Interaction`] — sim-side declaration of where a pawn can stand to
//! interact with a world object.
//!
//! Authored on the [`KindDef`] (RON), consumed by the path-finder. A tree's
//! interaction is "any of the 8 surrounding tiles" (`Surround`); a building
//! with a single door is `Facing(&[door_offset])` rotating with the
//! building's placement; an object that only accepts a specific world-aligned
//! cell (a campfire facing a specific bench) is `Offsets(&[...])`.
//!
//! ## Variants
//!
//! The variant *implies* whether the object's facing rotates the candidate
//! tiles, so callers don't carry a bool:
//!
//! - [`Interaction::None`] — not interactable (the implicit default for kinds
//!   that omit the field).
//! - [`Interaction::Surround { radius_tiles }`](Interaction::Surround) —
//!   every tile within Chebyshev distance `radius_tiles` of the object's
//!   origin tile, *excluding* the origin itself. Rotation-invariant.
//! - [`Interaction::Offsets`] — world-aligned tile offsets from the object's
//!   origin tile. Used when the canonical positions are tied to the world
//!   frame rather than to the object (e.g. a campfire whose seating is
//!   placed by the designer relative to the map).
//! - [`Interaction::Facing`] — tile offsets in the object's *local* frame,
//!   rotated by the object's [`Facing`] at query time. Used for placed
//!   buildings where the door / interaction cell rotates with the building.
//!
//! ## Authoring format
//!
//! The enum uses serde's internally-tagged representation with the discriminator
//! field `type`. This is one notch more verbose than serde's default
//! `Variant(payload)` form, but the default form does *not* round-trip through
//! the [`ron::Value`] that [`KindDef`] stores — variant tags get lost. Tagging
//! by a regular field-name (`type`) is a plain map and survives the
//! [`ron::Value`] step.
//!
//! ```ron
//! (
//!     id: "currawong:oak_tree",
//!     interaction: (type: "Surround", radius_tiles: 1),
//! )
//! ```
//!
//! Offsets are `(dx, dy, dz)` integer tile tuples:
//!
//! ```ron
//! interaction: (type: "Facing", offsets: [(1, 0, 0)])   // door one tile east of origin
//! interaction: (type: "Offsets", offsets: [(2, -1, 0)]) // world-aligned seating cell
//! ```
//!
//! Kinds that omit the `interaction:` field deserialize to [`Interaction::None`]
//! via serde's default.
//!
//! ## Resolving to world tiles
//!
//! [`Interaction::tiles`] takes the object's [`WorldTransform`] and returns
//! the candidate world tile coordinates. It does *not* check terrain
//! walkability — that's the path-finder's job. Interaction tiles are
//! *candidates*; the planner filters them against current passability when
//! it plans a path. Splitting "what the designer declared" from "what's
//! reachable right now" lets a tree against a wall fall back to its other 7
//! tiles without per-instance bookkeeping on the object.
//!
//! ## Facing rotation
//!
//! [`Interaction::Facing`] rotation snaps to the nearest cardinal-4 step
//! (0°/90°/180°/270°). Buildings only place at cardinal orientations, and
//! integer tile offsets only stay integer-valued under 90° rotations —
//! anything finer would either drift the tile off-grid or require fixed-point
//! sub-tile math the path-finder doesn't currently want.

use serde::Deserialize;

use crate::data::KindDef;

use super::facing::Facing;
use super::zone::WorldTransform;

/// Declared interaction positions for a kind. See the [module
/// docs](self) for variant semantics and authoring format.
///
/// [`Default`] is [`Interaction::None`], which is also what
/// [`Interaction::from_def`] returns for kinds that omit the `interaction:`
/// field.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(tag = "type")]
pub enum Interaction {
    /// Not interactable. The implicit default for kinds that omit the
    /// `interaction` field, but spelled explicitly here for kinds that want
    /// to make non-interactability part of their declared shape.
    #[default]
    None,
    /// Any tile within Chebyshev distance `radius_tiles` of the object's
    /// origin tile, excluding the origin itself. Rotation-invariant — the
    /// object's [`Facing`] doesn't affect the candidate set.
    ///
    /// Radius 1 (the common case for single-tile resources like trees) yields
    /// the 8 neighbouring tiles.
    Surround { radius_tiles: u8 },
    /// World-aligned tile offsets from the object's origin tile. Facing is
    /// ignored. Use this when the interaction cells are fixed relative to
    /// the world frame rather than to the object.
    Offsets { offsets: Vec<(i32, i32, i32)> },
    /// Tile offsets in the object's local frame, rotated by the object's
    /// [`Facing`] (snapped to cardinal-4) at query time. Use this for
    /// rotation-sensitive buildings — a door, an interaction surface that
    /// faces one way.
    Facing { offsets: Vec<(i32, i32, i32)> },
}

impl Interaction {
    /// World tile coordinates that satisfy this interaction for an object
    /// at `transform`. Returns an empty `Vec` for [`Interaction::None`].
    ///
    /// Tiles returned here are *candidates* — the path-finder is expected
    /// to filter for terrain walkability before treating any of them as a
    /// goal.
    pub fn tiles(&self, transform: &WorldTransform) -> Vec<(i32, i32, i32)> {
        let (ox, oy, oz) = transform.position.tile_coord();
        match self {
            Self::None => Vec::new(),
            Self::Surround { radius_tiles } => {
                let r = *radius_tiles as i32;
                let count = ((2 * r + 1) * (2 * r + 1) - 1).max(0) as usize;
                let mut out = Vec::with_capacity(count);
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        out.push((ox + dx, oy + dy, oz));
                    }
                }
                out
            }
            Self::Offsets { offsets } => offsets
                .iter()
                .map(|(dx, dy, dz)| (ox + dx, oy + dy, oz + dz))
                .collect(),
            Self::Facing { offsets } => {
                let quarter = rotate_quarter(transform.facing);
                offsets
                    .iter()
                    .map(|&(dx, dy, dz)| {
                        let (rx, ry) = rotate_xy(dx, dy, quarter);
                        (ox + rx, oy + ry, oz + dz)
                    })
                    .collect()
            }
        }
    }

    /// Pull the `interaction:` field out of a kind def's RON body. Kinds
    /// that omit the field deserialize to [`Interaction::None`] via serde's
    /// default; explicit `interaction: None` rounds to the same value. `Err`
    /// for malformed bodies.
    ///
    /// Mirrors the [`RenderSpec::from_def`](crate::render::RenderSpec)
    /// pattern — sim-side typed view onto an otherwise-untyped [`KindDef`].
    pub fn from_def(def: &KindDef) -> Result<Self, ron::Error> {
        #[derive(Deserialize)]
        struct Body {
            #[serde(default)]
            interaction: Interaction,
        }
        let body: Body = def.value.clone().into_rust()?;
        Ok(body.interaction)
    }
}

/// Round a [`Facing`] to the nearest cardinal-4 step: `0 = +X (E)`,
/// `1 = +Y (N)`, `2 = -X (W)`, `3 = -Y (S)`. Symmetric rounding (half-sector
/// offset) so every quadrant is centred on its cardinal direction.
#[inline]
fn rotate_quarter(facing: Facing) -> u8 {
    // FULL_TURN / 4 = 16384; half a quarter = 8192 = 1 << 13.
    let half_quarter = 1u32 << 13;
    (((facing.0 as u32).wrapping_add(half_quarter) >> 14) & 0b11) as u8
}

/// Rotate an integer (x, y) tile offset by `quarter` 90° steps
/// counterclockwise. Z is unaffected (facing is yaw-only).
#[inline]
fn rotate_xy(x: i32, y: i32, quarter: u8) -> (i32, i32) {
    match quarter & 0b11 {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        3 => (y, -x),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{KindId, VfsPath};
    use crate::sim::units::SimPos;

    fn at(x: i32, y: i32, z: i32) -> WorldTransform {
        WorldTransform {
            position: SimPos::tile(x, y, z),
            facing: Facing::ZERO,
        }
    }

    fn at_facing(x: i32, y: i32, z: i32, facing: Facing) -> WorldTransform {
        WorldTransform {
            position: SimPos::tile(x, y, z),
            facing,
        }
    }

    fn def_with_body(ron_text: &str) -> KindDef {
        let value: ron::Value = ron::from_str(ron_text).expect("test RON parses");
        KindDef {
            id: KindId::new("currawong:test").unwrap(),
            source: VfsPath::new("test.ron").unwrap(),
            value,
        }
    }

    // ----- None ------------------------------------------------------

    #[test]
    fn none_yields_no_tiles() {
        let tiles = Interaction::None.tiles(&at(5, 5, 0));
        assert!(tiles.is_empty());
    }

    // ----- Surround --------------------------------------------------

    #[test]
    fn surround_radius_1_yields_8_neighbours() {
        let tiles = Interaction::Surround { radius_tiles: 1 }.tiles(&at(0, 0, 0));
        assert_eq!(tiles.len(), 8);
        // Spot-check the cardinals and one diagonal.
        assert!(tiles.contains(&(1, 0, 0)));
        assert!(tiles.contains(&(-1, 0, 0)));
        assert!(tiles.contains(&(0, 1, 0)));
        assert!(tiles.contains(&(0, -1, 0)));
        assert!(tiles.contains(&(1, 1, 0)));
        // The origin itself is not a candidate.
        assert!(!tiles.contains(&(0, 0, 0)));
    }

    #[test]
    fn surround_translates_by_origin() {
        let tiles = Interaction::Surround { radius_tiles: 1 }.tiles(&at(10, 7, 2));
        assert_eq!(tiles.len(), 8);
        assert!(tiles.contains(&(11, 7, 2)));
        assert!(tiles.contains(&(9, 6, 2)));
        // Z is unchanged — surround is planar.
        for (_, _, z) in &tiles {
            assert_eq!(*z, 2);
        }
    }

    #[test]
    fn surround_ignores_facing() {
        let a = Interaction::Surround { radius_tiles: 1 }.tiles(&at(0, 0, 0));
        let b = Interaction::Surround { radius_tiles: 1 }.tiles(&at_facing(0, 0, 0, Facing::HALF));
        // Sets are equal regardless of facing.
        let mut a_sorted = a;
        let mut b_sorted = b;
        a_sorted.sort();
        b_sorted.sort();
        assert_eq!(a_sorted, b_sorted);
    }

    #[test]
    fn surround_radius_2_yields_24_tiles() {
        let tiles = Interaction::Surround { radius_tiles: 2 }.tiles(&at(0, 0, 0));
        // (2*2+1)^2 - 1 = 24.
        assert_eq!(tiles.len(), 24);
        assert!(tiles.contains(&(2, 2, 0)));
        assert!(tiles.contains(&(-2, 0, 0)));
        assert!(!tiles.contains(&(0, 0, 0)));
    }

    #[test]
    fn surround_radius_0_yields_no_tiles() {
        let tiles = Interaction::Surround { radius_tiles: 0 }.tiles(&at(0, 0, 0));
        assert!(tiles.is_empty());
    }

    // ----- Offsets (world-aligned) -----------------------------------

    #[test]
    fn offsets_are_world_aligned() {
        let interaction = Interaction::Offsets {
            offsets: vec![(1, 0, 0), (0, 2, -1)],
        };
        let here = interaction.tiles(&at(5, 5, 0));
        assert_eq!(here, vec![(6, 5, 0), (5, 7, -1)]);
    }

    #[test]
    fn offsets_ignore_facing() {
        let interaction = Interaction::Offsets {
            offsets: vec![(1, 0, 0)],
        };
        let a = interaction.tiles(&at_facing(0, 0, 0, Facing::ZERO));
        let b = interaction.tiles(&at_facing(0, 0, 0, Facing::QUARTER));
        assert_eq!(a, b);
        assert_eq!(a, vec![(1, 0, 0)]);
    }

    // ----- Facing (rotated with object) ------------------------------

    #[test]
    fn facing_zero_does_not_rotate() {
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, Facing::ZERO));
        assert_eq!(tiles, vec![(1, 0, 0)]);
    }

    #[test]
    fn facing_quarter_rotates_x_to_y() {
        // Door offset is one tile east in local frame. Building rotated 90°
        // CCW means the door is now one tile north in world.
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, Facing::QUARTER));
        assert_eq!(tiles, vec![(0, 1, 0)]);
    }

    #[test]
    fn facing_half_rotates_x_to_negative_x() {
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, Facing::HALF));
        assert_eq!(tiles, vec![(-1, 0, 0)]);
    }

    #[test]
    fn facing_three_quarters_rotates_x_to_negative_y() {
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, Facing::THREE_QUARTERS));
        assert_eq!(tiles, vec![(0, -1, 0)]);
    }

    #[test]
    fn facing_preserves_z_offset() {
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 3)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, Facing::QUARTER));
        assert_eq!(tiles, vec![(0, 1, 3)]);
    }

    #[test]
    fn facing_snaps_off_axis_to_nearest_cardinal() {
        // 5° off from QUARTER (which is 90°). One unit ≈ 0.0055°, so 5°
        // ≈ 910 Facing units past QUARTER. Should still snap to quarter=1.
        let near_quarter = Facing(Facing::QUARTER.0 + 910);
        let interaction = Interaction::Facing {
            offsets: vec![(1, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(0, 0, 0, near_quarter));
        assert_eq!(tiles, vec![(0, 1, 0)]);
    }

    #[test]
    fn facing_rotates_translates_with_origin() {
        // 3x3 building at (10, 10) facing +Y, door offset (2, 0, 0) in
        // local frame → world door at (10, 12, 0).
        let interaction = Interaction::Facing {
            offsets: vec![(2, 0, 0)],
        };
        let tiles = interaction.tiles(&at_facing(10, 10, 0, Facing::QUARTER));
        assert_eq!(tiles, vec![(10, 12, 0)]);
    }

    // ----- from_def --------------------------------------------------

    #[test]
    fn from_def_surround() {
        let def = def_with_body(
            r#"(
                id: "currawong:oak_tree",
                interaction: (type: "Surround", radius_tiles: 1),
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(parsed, Interaction::Surround { radius_tiles: 1 });
    }

    #[test]
    fn from_def_facing_with_offsets() {
        let def = def_with_body(
            r#"(
                id: "currawong:lumber_camp",
                interaction: (type: "Facing", offsets: [(1, 0, 0), (1, 1, 0)]),
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(
            parsed,
            Interaction::Facing {
                offsets: vec![(1, 0, 0), (1, 1, 0)],
            }
        );
    }

    #[test]
    fn from_def_offsets() {
        let def = def_with_body(
            r#"(
                id: "currawong:campfire",
                interaction: (type: "Offsets", offsets: [(2, -1, 0)]),
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(
            parsed,
            Interaction::Offsets {
                offsets: vec![(2, -1, 0)],
            }
        );
    }

    #[test]
    fn from_def_none_variant() {
        let def = def_with_body(
            r#"(
                id: "currawong:rock",
                interaction: (type: "None"),
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(parsed, Interaction::None);
    }

    #[test]
    fn from_def_missing_field_defaults_to_none() {
        // No `interaction` field — the implicit-non-interactable case
        // collapses to `Interaction::None` via serde's default. Consumers
        // don't need to distinguish "explicitly None" from "missing".
        let def = def_with_body(
            r#"(
                id: "currawong:plain_kind",
                chop_ticks: 90,
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(parsed, Interaction::None);
    }

    #[test]
    fn from_def_coexists_with_other_fields() {
        // Real kind defs carry `render`, `chop_ticks`, etc. The parse must
        // ignore them — it only cares about `interaction`.
        let def = def_with_body(
            r#"(
                id: "currawong:oak_tree",
                chop_ticks: 90,
                interaction: (type: "Surround", radius_tiles: 1),
                render: (
                    shape: "tree",
                    mesh: "trees/oak.glb",
                ),
            )"#,
        );
        let parsed = Interaction::from_def(&def).unwrap();
        assert_eq!(parsed, Interaction::Surround { radius_tiles: 1 });
    }

    #[test]
    fn from_def_malformed_surfaces_error() {
        let def = def_with_body(
            r#"(
                id: "currawong:bad",
                interaction: (type: "Surround", radius_tiles: "not a number"),
            )"#,
        );
        assert!(Interaction::from_def(&def).is_err());
    }
}
