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
/// and (forthcoming) `HexGrid` (6/6). Game code picks one when constructing
/// `Terrain<G>`; the engine never mixes grids within a zone.
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
}
