//! [`Footprint`] — sim-side declaration of the ground tiles an object
//! occupies when placed.
//!
//! Authored on the [`KindDef`] (RON) as a flat `footprint:` array of
//! `(dx, dy)` pairs relative to the object's origin tile. A pawn with no
//! declared footprint defaults to empty (the kind isn't tile-occupying);
//! a 1×1 tree declares `[(0, 0)]`; a 2×2 building declares all four tiles.
//!
//! Distinct from [`Interaction`](super::Interaction): footprint says
//! *which tiles the object sits on*; interaction says *where a pawn can
//! stand to interact with the object*. Together they describe a placed
//! object's spatial relationship to the tile grid.
//!
//! ## Authoring format
//!
//! ```ron
//! (
//!     id: "currawong:lumber_camp",
//!     footprint: [(0, 0), (1, 0), (0, 1), (1, 1)],
//! )
//! ```
//!
//! 2D pairs only — footprint is planar; the Z coordinate of every tile is
//! taken from the object's `WorldTransform.position` at resolve time.
//! Kinds that omit the `footprint:` field deserialize to an empty footprint
//! via serde's default — the overlay simply draws zero instances.
//!
//! ## Resolving to world tiles
//!
//! [`Footprint::tiles`] takes the object's [`WorldTransform`] and returns
//! the world tile coordinates the footprint covers. Offsets are
//! world-aligned (not rotated by [`Facing`]) — the current consumers
//! place buildings cardinal-aligned, and rotating a multi-tile footprint
//! cleanly under arbitrary facings is a follow-up if/when a rotating
//! placement system actually needs it (same as how [`Interaction`]
//! started world-aligned with [`Offsets`](super::Interaction::Offsets)
//! and added [`Facing`](super::Interaction::Facing) later).

use serde::Deserialize;

use crate::data::KindDef;

use super::zone::WorldTransform;

/// Tile offsets that make up an object's placement footprint. World-aligned
/// 2D offsets from the object's origin tile; see the [module docs](self)
/// for authoring format.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct Footprint(pub Vec<(i32, i32)>);

impl Footprint {
    /// World tile coordinates this footprint covers for an object at
    /// `transform`. Z is taken from the object's origin tile — footprint
    /// is planar.
    pub fn tiles(&self, transform: &WorldTransform) -> Vec<(i32, i32, i32)> {
        let (ox, oy, oz) = transform.position.tile_coord();
        self.0
            .iter()
            .map(|(dx, dy)| (ox + dx, oy + dy, oz))
            .collect()
    }

    /// Pull the `footprint:` field out of a kind def's RON body. Kinds that
    /// omit the field deserialize to an empty footprint via serde's
    /// default. `Err` for malformed bodies.
    pub fn from_def(def: &KindDef) -> Result<Self, ron::Error> {
        #[derive(Deserialize)]
        struct Body {
            #[serde(default)]
            footprint: Footprint,
        }
        let body: Body = def.value.clone().into_rust()?;
        Ok(body.footprint)
    }

    /// `true` when no tiles are declared (the default for kinds that omit
    /// the `footprint:` field).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{KindId, VfsPath};
    use crate::sim::facing::Facing;
    use crate::sim::units::SimPos;

    fn at(x: i32, y: i32, z: i32) -> WorldTransform {
        WorldTransform {
            position: SimPos::tile(x, y, z),
            facing: Facing::ZERO,
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

    #[test]
    fn empty_footprint_yields_no_tiles() {
        let tiles = Footprint::default().tiles(&at(5, 5, 0));
        assert!(tiles.is_empty());
    }

    #[test]
    fn single_tile_footprint() {
        let footprint = Footprint(vec![(0, 0)]);
        let tiles = footprint.tiles(&at(5, 7, 2));
        assert_eq!(tiles, vec![(5, 7, 2)]);
    }

    #[test]
    fn two_by_two_translates_by_origin() {
        let footprint = Footprint(vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        let tiles = footprint.tiles(&at(10, 10, 0));
        assert_eq!(
            tiles,
            vec![(10, 10, 0), (11, 10, 0), (10, 11, 0), (11, 11, 0)]
        );
    }

    #[test]
    fn z_taken_from_origin() {
        let footprint = Footprint(vec![(0, 0), (1, 0)]);
        let tiles = footprint.tiles(&at(0, 0, 3));
        for (_, _, z) in &tiles {
            assert_eq!(*z, 3);
        }
    }

    #[test]
    fn from_def_basic() {
        let def = def_with_body(
            r#"(
                id: "currawong:lumber_camp",
                footprint: [(0, 0), (1, 0), (0, 1), (1, 1)],
            )"#,
        );
        let parsed = Footprint::from_def(&def).unwrap();
        assert_eq!(parsed, Footprint(vec![(0, 0), (1, 0), (0, 1), (1, 1)]),);
    }

    #[test]
    fn from_def_single_tile() {
        let def = def_with_body(
            r#"(
                id: "currawong:torch",
                footprint: [(0, 0)],
            )"#,
        );
        let parsed = Footprint::from_def(&def).unwrap();
        assert_eq!(parsed, Footprint(vec![(0, 0)]));
    }

    #[test]
    fn from_def_missing_field_defaults_to_empty() {
        let def = def_with_body(
            r#"(
                id: "currawong:plain_kind",
                chop_ticks: 90,
            )"#,
        );
        let parsed = Footprint::from_def(&def).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn from_def_coexists_with_other_fields() {
        let def = def_with_body(
            r#"(
                id: "currawong:lumber_camp",
                interaction: (type: "Facing", offsets: [(-1, 0, 0)]),
                footprint: [(0, 0), (1, 0)],
                render: (
                    shape: "building",
                    mesh: "lumber/base.glb",
                ),
            )"#,
        );
        let parsed = Footprint::from_def(&def).unwrap();
        assert_eq!(parsed, Footprint(vec![(0, 0), (1, 0)]));
    }

    #[test]
    fn from_def_malformed_surfaces_error() {
        let def = def_with_body(
            r#"(
                id: "currawong:bad",
                footprint: [(0, "not a number")],
            )"#,
        );
        assert!(Footprint::from_def(&def).is_err());
    }
}
