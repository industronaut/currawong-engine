//! Grid topology — the shape of a zone's tile lattice.
//!
//! [`Terrain`](super::Terrain) is generic over [`Grid`]: the same chunk data,
//! the same mesher, work over both square and hex grids by delegating "where
//! does a cell sit, what are its corners, who are its neighbours" to the grid
//! impl. Topology only — scale (world units per cell) lives on the mesher.
//!
//! Cells are addressed by [`IVec2`] in both impls (axial coords for hex), so
//! the chunk storage doesn't need to know which grid it sits in.
//!
//! ## Conventions
//!
//! - Corners are indexed `0..CORNERS_PER_CELL`, ordered CCW when viewed from
//!   +Z. Edges share that ordering: edge `i` runs between corner `i` and
//!   corner `(i + 1) % CORNERS_PER_CELL`. [`Grid::neighbour`] across edge `i`
//!   is the cell on the far side of that edge.
//! - Positions returned by [`Grid::cell_center`] and [`Grid::corner_xy`] are
//!   in *canonical unit space*: square cells have unit width, hex cells have
//!   unit edge length. A mesher applies its own `tile_size` to scale into
//!   world space.

use glam::{IVec2, Vec2};

/// Topology of a tile lattice. Used by [`Terrain`](super::Terrain) and
/// view-side meshers to walk cells, find neighbours, and place corners.
///
/// Two impls live in this module: [`SquareGrid`] (4 neighbours, 4 corners)
/// and [`HexGrid`] (6/6, flat-top axial). Game code picks one when
/// constructing `Terrain<G>`; the engine never mixes grids within a zone.
pub trait Grid {
    /// Number of corners on one cell. 4 for square, 6 for hex.
    const CORNERS_PER_CELL: usize;

    /// Number of edges on one cell. Equal to [`Self::CORNERS_PER_CELL`] for
    /// both regular grids; tracked separately so future irregular impls can
    /// diverge if they ever land.
    const EDGES_PER_CELL: usize;

    /// Centre of the cell in canonical unit space.
    fn cell_center(&self, cell: IVec2) -> Vec2;

    /// Position of one of the cell's corners in canonical unit space.
    /// Corners are ordered CCW from above; index in `0..CORNERS_PER_CELL`.
    fn corner_xy(&self, cell: IVec2, corner_idx: usize) -> Vec2;

    /// Neighbour cell across edge `edge_idx`. Edge `i` runs between corner
    /// `i` and corner `(i + 1) % CORNERS_PER_CELL`. Index in `0..EDGES_PER_CELL`.
    fn neighbour(&self, cell: IVec2, edge_idx: usize) -> IVec2;

    /// Every cell that touches corner `corner_idx` of `cell`, *including*
    /// `cell` itself. Used by the slope mesher to compute per-corner heights
    /// as the max of touching cells' `floor_height`.
    ///
    /// For square: 4 cells at interior corners (self + two edge neighbours
    /// flanking the corner + the diagonal at their intersection).
    /// For hex: 3 cells (self + the two edge neighbours flanking the corner).
    fn cells_at_corner(&self, cell: IVec2, corner_idx: usize) -> impl Iterator<Item = IVec2>;

    /// Inverse of [`cell_center`](Self::cell_center): the cell whose footprint
    /// contains `point` in canonical unit space. Total — every point belongs
    /// to exactly one cell, even points exactly on a shared edge (each impl
    /// picks a deterministic side).
    ///
    /// View-side picking uses this: ray vs. ground plane gives a world XY,
    /// divide out the mesher's `tile_size` to land in canonical space, then
    /// ask the grid which cell the cursor is over.
    fn cell_at(&self, point: Vec2) -> IVec2;
}

// --- SquareGrid ----------------------------------------------------------

/// Axis-aligned unit-square grid. Cell `(tx, ty)` covers
/// `[tx, tx+1) × [ty, ty+1)` in canonical unit space.
///
/// Corner ordering CCW from above, starting at the corner shared by the
/// `+X` and `-Y` edges:
///
/// ```text
///   corner 2 ─── corner 1
///       │           │
///   edge 2       edge 0
///       │           │
///   corner 3 ─── corner 0
///         edge 3
/// ```
///
/// Edge `i` borders the neighbour cell:
/// - 0: `+X` (`(tx+1, ty)`)
/// - 1: `+Y` (`(tx, ty+1)`)
/// - 2: `-X` (`(tx-1, ty)`)
/// - 3: `-Y` (`(tx, ty-1)`)
#[derive(Debug, Default, Clone, Copy)]
pub struct SquareGrid;

impl Grid for SquareGrid {
    const CORNERS_PER_CELL: usize = 4;
    const EDGES_PER_CELL: usize = 4;

    fn cell_center(&self, cell: IVec2) -> Vec2 {
        Vec2::new(cell.x as f32 + 0.5, cell.y as f32 + 0.5)
    }

    fn corner_xy(&self, cell: IVec2, corner_idx: usize) -> Vec2 {
        let (dx, dy) = match corner_idx {
            0 => (1, 0),
            1 => (1, 1),
            2 => (0, 1),
            3 => (0, 0),
            _ => panic!("SquareGrid::corner_xy: corner_idx must be in 0..4, got {corner_idx}",),
        };
        Vec2::new((cell.x + dx) as f32, (cell.y + dy) as f32)
    }

    fn neighbour(&self, cell: IVec2, edge_idx: usize) -> IVec2 {
        let delta = match edge_idx {
            0 => IVec2::new(1, 0),
            1 => IVec2::new(0, 1),
            2 => IVec2::new(-1, 0),
            3 => IVec2::new(0, -1),
            _ => panic!("SquareGrid::neighbour: edge_idx must be in 0..4, got {edge_idx}",),
        };
        cell + delta
    }

    fn cell_at(&self, point: Vec2) -> IVec2 {
        // Cell (tx, ty) covers [tx, tx+1) × [ty, ty+1), so floor maps every
        // point unambiguously: a point exactly on `x = tx` belongs to cell
        // `tx`, not `tx - 1`.
        IVec2::new(point.x.floor() as i32, point.y.floor() as i32)
    }

    fn cells_at_corner(&self, cell: IVec2, corner_idx: usize) -> impl Iterator<Item = IVec2> {
        // Corner `i` sits at the meeting of edges `(i+3) % 4` and `i`. The
        // four cells touching it are: self, the edge-`i` neighbour, the
        // edge-`(i+3) % 4` neighbour, and the diagonal at their corner.
        let deltas = match corner_idx {
            0 => [
                IVec2::new(0, 0),
                IVec2::new(1, 0),
                IVec2::new(1, -1),
                IVec2::new(0, -1),
            ],
            1 => [
                IVec2::new(0, 0),
                IVec2::new(1, 0),
                IVec2::new(1, 1),
                IVec2::new(0, 1),
            ],
            2 => [
                IVec2::new(0, 0),
                IVec2::new(-1, 0),
                IVec2::new(-1, 1),
                IVec2::new(0, 1),
            ],
            3 => [
                IVec2::new(0, 0),
                IVec2::new(-1, 0),
                IVec2::new(-1, -1),
                IVec2::new(0, -1),
            ],
            _ => {
                panic!("SquareGrid::cells_at_corner: corner_idx must be in 0..4, got {corner_idx}")
            }
        };
        deltas.into_iter().map(move |d| cell + d)
    }
}

// --- HexGrid -------------------------------------------------------------

/// Flat-top hex grid in axial coordinates `(q, r)`, stored as
/// `IVec2 { x: q, y: r }`. Edge length is 1 in canonical unit space — the
/// mesher scales by its own `tile_size` to reach world units.
///
/// Layout: `+q` columns step right and up; `+r` rows step straight up.
/// Concretely, `cell_center(q, r) = (1.5·q,  (√3/2)·q + √3·r)`. With this
/// axial convention each hex has flat edges at top (between corners 1 and
/// 2) and bottom (between corners 4 and 5), and pointed tips at east
/// (corner 0) and west (corner 3).
///
/// Corner ordering, CCW from the East tip:
///
/// ```text
///           corner 2 ─── corner 1
///              /             \
///         edge 2             edge 0
///            /                 \
///        corner 3              corner 0   (East tip)
///            \                 /
///         edge 3             edge 5
///              \             /
///           corner 4 ─── corner 5
/// ```
///
/// Edge `i` is between corner `i` and corner `(i + 1) % 6`, and is shared
/// with the neighbour at axial offset:
/// - 0 (NE side):    `(+1,  0)`
/// - 1 (N flat top): `( 0, +1)`
/// - 2 (NW side):    `(-1, +1)`
/// - 3 (SW side):    `(-1,  0)`
/// - 4 (S flat bot): `( 0, -1)`
/// - 5 (SE side):    `(+1, -1)`
#[derive(Debug, Default, Clone, Copy)]
pub struct HexGrid;

/// Precomputed `√3 / 2`. Used for hex corner offsets and centre math.
const SQRT3_HALF: f32 = 0.866_025_4;
/// Precomputed `√3`. Vertical spacing between hex rows in canonical space.
const SQRT3: f32 = 1.732_050_8;

/// Corner offsets from the hex centre at unit edge length, CCW from East.
const HEX_CORNER_OFFSETS: [Vec2; 6] = [
    Vec2::new(1.0, 0.0),
    Vec2::new(0.5, SQRT3_HALF),
    Vec2::new(-0.5, SQRT3_HALF),
    Vec2::new(-1.0, 0.0),
    Vec2::new(-0.5, -SQRT3_HALF),
    Vec2::new(0.5, -SQRT3_HALF),
];

/// Axial-coordinate offsets to the six edge neighbours, indexed by
/// [`Grid::neighbour`]'s `edge_idx`.
const HEX_NEIGHBOUR_OFFSETS: [IVec2; 6] = [
    IVec2::new(1, 0),
    IVec2::new(0, 1),
    IVec2::new(-1, 1),
    IVec2::new(-1, 0),
    IVec2::new(0, -1),
    IVec2::new(1, -1),
];

impl Grid for HexGrid {
    const CORNERS_PER_CELL: usize = 6;
    const EDGES_PER_CELL: usize = 6;

    fn cell_center(&self, cell: IVec2) -> Vec2 {
        let q = cell.x as f32;
        let r = cell.y as f32;
        Vec2::new(1.5 * q, SQRT3_HALF * q + SQRT3 * r)
    }

    fn corner_xy(&self, cell: IVec2, corner_idx: usize) -> Vec2 {
        let offset = HEX_CORNER_OFFSETS
            .get(corner_idx)
            .copied()
            .unwrap_or_else(|| {
                panic!("HexGrid::corner_xy: corner_idx must be in 0..6, got {corner_idx}")
            });
        self.cell_center(cell) + offset
    }

    fn neighbour(&self, cell: IVec2, edge_idx: usize) -> IVec2 {
        let delta = HEX_NEIGHBOUR_OFFSETS
            .get(edge_idx)
            .copied()
            .unwrap_or_else(|| {
                panic!("HexGrid::neighbour: edge_idx must be in 0..6, got {edge_idx}")
            });
        cell + delta
    }

    fn cell_at(&self, point: Vec2) -> IVec2 {
        // Inverse of `cell_center`: x = 1.5q and y = (√3/2)q + √3 r, so
        //   q = (2/3) x
        //   r = y / √3 − x / 3
        // gives fractional axial coordinates. The classical fix-up is to go
        // through cube coordinates (q, r, s) with s = −q − r, round each,
        // then re-impose s = −q − r by snapping back the axis with the
        // largest rounding error. Same trick as Red Blob Games' hex rounding.
        let q_frac = (2.0 / 3.0) * point.x;
        let r_frac = point.y / SQRT3 - point.x / 3.0;
        let s_frac = -q_frac - r_frac;

        let mut q = q_frac.round();
        let mut r = r_frac.round();
        let s = s_frac.round();

        let dq = (q - q_frac).abs();
        let dr = (r - r_frac).abs();
        let ds = (s - s_frac).abs();

        if dq > dr && dq > ds {
            q = -r - s;
        } else if dr > ds {
            r = -q - s;
        }
        // `else` branch would snap `s`, but we don't carry s in axial output.

        IVec2::new(q as i32, r as i32)
    }

    fn cells_at_corner(&self, cell: IVec2, corner_idx: usize) -> impl Iterator<Item = IVec2> {
        // Corner `i` sits at the meeting of edges `(i+5) % 6` (incoming from
        // corner i-1) and `i` (outgoing to corner i+1). The three cells
        // touching it are self plus the two edge neighbours flanking it.
        assert!(
            corner_idx < HexGrid::CORNERS_PER_CELL,
            "HexGrid::cells_at_corner: corner_idx must be in 0..6, got {corner_idx}",
        );
        let next_edge = corner_idx;
        let prev_edge = (corner_idx + 5) % 6;
        [
            cell,
            cell + HEX_NEIGHBOUR_OFFSETS[next_edge],
            cell + HEX_NEIGHBOUR_OFFSETS[prev_edge],
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_corner_ccw_from_plus_x_minus_y() {
        // Sanity-check the corner order against the doc comment.
        let g = SquareGrid;
        let c = IVec2::new(3, 5);
        assert_eq!(g.corner_xy(c, 0), Vec2::new(4.0, 5.0));
        assert_eq!(g.corner_xy(c, 1), Vec2::new(4.0, 6.0));
        assert_eq!(g.corner_xy(c, 2), Vec2::new(3.0, 6.0));
        assert_eq!(g.corner_xy(c, 3), Vec2::new(3.0, 5.0));
    }

    #[test]
    fn square_neighbour_matches_edge_index() {
        let g = SquareGrid;
        let c = IVec2::new(0, 0);
        assert_eq!(g.neighbour(c, 0), IVec2::new(1, 0));
        assert_eq!(g.neighbour(c, 1), IVec2::new(0, 1));
        assert_eq!(g.neighbour(c, 2), IVec2::new(-1, 0));
        assert_eq!(g.neighbour(c, 3), IVec2::new(0, -1));
    }

    #[test]
    fn square_edge_bounded_by_consecutive_corners() {
        // The invariant the mesher relies on: edge `i` runs between
        // corner `i` and corner `(i+1) % CORNERS_PER_CELL`, and the edge's
        // midpoint lies on the boundary between the cell and its neighbour
        // across that edge.
        let g = SquareGrid;
        let c = IVec2::new(0, 0);
        for edge in 0..SquareGrid::EDGES_PER_CELL {
            let c0 = g.corner_xy(c, edge);
            let c1 = g.corner_xy(c, (edge + 1) % SquareGrid::CORNERS_PER_CELL);
            let edge_mid = (c0 + c1) * 0.5;
            let neighbour = g.neighbour(c, edge);
            let neighbour_center = g.cell_center(neighbour);
            let own_center = g.cell_center(c);
            let halfway = (own_center + neighbour_center) * 0.5;
            assert!(
                (edge_mid - halfway).length() < 1e-6,
                "edge {edge}: midpoint {edge_mid:?} != halfway-to-neighbour {halfway:?}",
            );
        }
    }

    #[test]
    fn square_cell_center_is_unit_offset() {
        let g = SquareGrid;
        assert_eq!(g.cell_center(IVec2::new(0, 0)), Vec2::new(0.5, 0.5));
        assert_eq!(g.cell_center(IVec2::new(-1, 2)), Vec2::new(-0.5, 2.5));
    }

    // --- HexGrid ----------------------------------------------------------

    fn approx_eq(a: Vec2, b: Vec2, eps: f32) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn hex_origin_corners_form_unit_ring() {
        // Every corner of the origin hex is at distance 1 from its centre
        // (canonical edge length = 1). This double-checks the lookup table
        // wasn't mistyped.
        let g = HexGrid;
        let c = IVec2::ZERO;
        for i in 0..HexGrid::CORNERS_PER_CELL {
            let r = (g.corner_xy(c, i) - g.cell_center(c)).length();
            assert!((r - 1.0).abs() < 1e-6, "corner {i} radius {r} != 1.0");
        }
    }

    #[test]
    fn hex_corner_0_is_east_corner_3_is_west() {
        // Pin the "flat-top, East tip at corner 0" orientation so the rest
        // of the trait surface — and the mesher's wall winding — has a
        // stable starting point.
        let g = HexGrid;
        let c = g.cell_center(IVec2::ZERO);
        let east = g.corner_xy(IVec2::ZERO, 0);
        let west = g.corner_xy(IVec2::ZERO, 3);
        assert!(
            east.x > c.x && (east.y - c.y).abs() < 1e-6,
            "corner 0 = East"
        );
        assert!(
            west.x < c.x && (west.y - c.y).abs() < 1e-6,
            "corner 3 = West"
        );
    }

    #[test]
    fn hex_neighbour_matches_edge_index() {
        let g = HexGrid;
        let c = IVec2::new(0, 0);
        assert_eq!(g.neighbour(c, 0), IVec2::new(1, 0));
        assert_eq!(g.neighbour(c, 1), IVec2::new(0, 1));
        assert_eq!(g.neighbour(c, 2), IVec2::new(-1, 1));
        assert_eq!(g.neighbour(c, 3), IVec2::new(-1, 0));
        assert_eq!(g.neighbour(c, 4), IVec2::new(0, -1));
        assert_eq!(g.neighbour(c, 5), IVec2::new(1, -1));
    }

    #[test]
    fn hex_edge_bounded_by_consecutive_corners() {
        // Same invariant as `square_edge_bounded_by_consecutive_corners`,
        // pressure-tested on a different cell so off-origin offsets don't
        // sneak in a sign error.
        let g = HexGrid;
        let c = IVec2::new(2, -1);
        for edge in 0..HexGrid::EDGES_PER_CELL {
            let c0 = g.corner_xy(c, edge);
            let c1 = g.corner_xy(c, (edge + 1) % HexGrid::CORNERS_PER_CELL);
            let edge_mid = (c0 + c1) * 0.5;
            let neighbour = g.neighbour(c, edge);
            let halfway = (g.cell_center(c) + g.cell_center(neighbour)) * 0.5;
            assert!(
                approx_eq(edge_mid, halfway, 1e-5),
                "edge {edge}: midpoint {edge_mid:?} != halfway-to-neighbour {halfway:?}",
            );
        }
    }

    #[test]
    fn hex_neighbour_symmetric() {
        // If `a`'s neighbour across edge `i` is `b`, then `b`'s neighbour
        // across edge `(i + 3) % 6` must be `a`. Catches off-by-one in the
        // neighbour table.
        let g = HexGrid;
        let a = IVec2::new(3, -2);
        for i in 0..HexGrid::EDGES_PER_CELL {
            let b = g.neighbour(a, i);
            let back = g.neighbour(b, (i + 3) % HexGrid::EDGES_PER_CELL);
            assert_eq!(back, a, "edge {i}: {a:?} → {b:?} → {back:?} != {a:?}");
        }
    }

    #[test]
    fn hex_row_step_is_sqrt3_vertical() {
        // Stepping by (0, +1) in axial coords should move straight up by √3
        // in canonical space — the canonical row spacing for flat-top hex.
        let g = HexGrid;
        let a = g.cell_center(IVec2::new(0, 0));
        let b = g.cell_center(IVec2::new(0, 1));
        assert!((b.x - a.x).abs() < 1e-6);
        assert!((b.y - a.y - SQRT3).abs() < 1e-6);
    }

    // --- cells_at_corner --------------------------------------------------

    /// The invariant the slope mesher relies on: every cell in
    /// `cells_at_corner(cell, i)` must list `cell` in *its own*
    /// `cells_at_corner` for some matching corner. Without this, adjacent
    /// cells would compute different "max heights" for what's physically the
    /// same corner.
    fn corner_sharing_symmetric<G: Grid>(grid: &G, cell: IVec2) {
        for corner in 0..G::CORNERS_PER_CELL {
            let own_pos = grid.corner_xy(cell, corner);
            for other in grid.cells_at_corner(cell, corner) {
                let found = (0..G::CORNERS_PER_CELL)
                    .map(|i| grid.corner_xy(other, i))
                    .any(|p| (p - own_pos).length() < 1e-5)
                    && grid
                        .cells_at_corner(other, 0)
                        .chain(
                            (1..G::CORNERS_PER_CELL).flat_map(|i| grid.cells_at_corner(other, i)),
                        )
                        .any(|c| c == cell);
                assert!(
                    found,
                    "corner {corner} of {cell:?} touches {other:?}, but {other:?} \
                     doesn't list {cell:?} as a corner-neighbour",
                );
            }
        }
    }

    #[test]
    fn square_cells_at_interior_corner_is_four() {
        let g = SquareGrid;
        for corner in 0..SquareGrid::CORNERS_PER_CELL {
            let cells: Vec<_> = g.cells_at_corner(IVec2::ZERO, corner).collect();
            assert_eq!(cells.len(), 4, "corner {corner}");
            assert!(cells.contains(&IVec2::ZERO), "self always present");
            // All four cells must agree on the corner position.
            let target = g.corner_xy(IVec2::ZERO, corner);
            for c in &cells {
                let agrees = (0..SquareGrid::CORNERS_PER_CELL)
                    .map(|i| g.corner_xy(*c, i))
                    .any(|p| (p - target).length() < 1e-6);
                assert!(agrees, "cell {c:?} doesn't have a corner at {target:?}");
            }
        }
    }

    #[test]
    fn square_cells_at_corner_symmetric() {
        corner_sharing_symmetric(&SquareGrid, IVec2::new(2, -3));
    }

    #[test]
    fn square_cell_at_roundtrips_centers() {
        // The centre of every cell must hash back to that cell. Catches
        // off-by-one or sign mistakes in the floor mapping.
        let g = SquareGrid;
        for ty in -5..5 {
            for tx in -5..5 {
                let cell = IVec2::new(tx, ty);
                assert_eq!(g.cell_at(g.cell_center(cell)), cell);
            }
        }
    }

    #[test]
    fn square_cell_at_left_edge_owned_by_higher_cell() {
        // Convention: a point on x = tx belongs to cell tx, not tx - 1.
        // Mirrors `[tx, tx+1) × [ty, ty+1)` semi-open ranges.
        let g = SquareGrid;
        assert_eq!(g.cell_at(Vec2::new(0.0, 0.0)), IVec2::new(0, 0));
        assert_eq!(g.cell_at(Vec2::new(-0.0001, 0.0)), IVec2::new(-1, 0));
        assert_eq!(g.cell_at(Vec2::new(2.0, -3.0)), IVec2::new(2, -3));
    }

    #[test]
    fn hex_cells_at_corner_is_three() {
        let g = HexGrid;
        for corner in 0..HexGrid::CORNERS_PER_CELL {
            let cells: Vec<_> = g.cells_at_corner(IVec2::ZERO, corner).collect();
            assert_eq!(cells.len(), 3, "corner {corner}");
            assert!(cells.contains(&IVec2::ZERO), "self always present");
            // All three cells must agree on the corner position.
            let target = g.corner_xy(IVec2::ZERO, corner);
            for c in &cells {
                let agrees = (0..HexGrid::CORNERS_PER_CELL)
                    .map(|i| g.corner_xy(*c, i))
                    .any(|p| (p - target).length() < 1e-5);
                assert!(agrees, "cell {c:?} doesn't have a corner at {target:?}");
            }
        }
    }

    #[test]
    fn hex_cells_at_corner_symmetric() {
        corner_sharing_symmetric(&HexGrid, IVec2::new(2, -3));
    }

    #[test]
    fn hex_cell_at_roundtrips_centers() {
        // Same correctness pin as the square version, on a range that covers
        // both axes and crosses the origin.
        let g = HexGrid;
        for r in -5..5 {
            for q in -5..5 {
                let cell = IVec2::new(q, r);
                assert_eq!(g.cell_at(g.cell_center(cell)), cell);
            }
        }
    }

    #[test]
    fn hex_cell_at_near_centre_stays_in_centre_cell() {
        // Small perturbations from a hex's centre stay inside that hex —
        // exercises the cube-rounding fix-up path. Step well below the
        // canonical edge length of 1.
        let g = HexGrid;
        let cell = IVec2::new(2, -1);
        let centre = g.cell_center(cell);
        for (dx, dy) in [(0.2, 0.0), (-0.2, 0.0), (0.0, 0.3), (0.0, -0.3)] {
            assert_eq!(
                g.cell_at(centre + Vec2::new(dx, dy)),
                cell,
                "dx={dx} dy={dy}"
            );
        }
    }

    #[test]
    fn hex_cell_at_corner_picks_one_neighbour_deterministically() {
        // A point exactly at a shared corner sits equidistant from three
        // hexes; the cube-rounding fix-up must still pick exactly one.
        // We don't pin *which* one — just that the result is stable across
        // calls and is one of the three sharing cells.
        let g = HexGrid;
        let cell = IVec2::ZERO;
        for corner in 0..HexGrid::CORNERS_PER_CELL {
            let p = g.corner_xy(cell, corner);
            let picked = g.cell_at(p);
            let candidates: Vec<_> = g.cells_at_corner(cell, corner).collect();
            assert!(
                candidates.contains(&picked),
                "corner {corner}: picked {picked:?} not in {candidates:?}",
            );
            // Determinism: same input twice → same output.
            assert_eq!(g.cell_at(p), picked);
        }
    }
}
