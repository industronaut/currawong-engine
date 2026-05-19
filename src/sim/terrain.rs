//! Tile-grid terrain with optional liquid per tile.
//!
//! Terrain is a [`Zone`](super::Zone)-level slot, separate from the
//! [`WorldTransform`](super::WorldTransform) slot-map: tiles are dense and uniform
//! where world objects are sparse and individuated, so they want different
//! storage shapes. Pathfinding and gameplay read tile data directly; the
//! view-side mesher (in `render::terrain`, not yet built) consumes the same
//! data to produce chunk meshes.
//!
//! ## Model
//!
//! - **Chunked, sparse.** [`Terrain`] holds a `HashMap<ChunkCoord, Chunk>`.
//!   Chunks are allocated lazily on write, so empty regions cost nothing and
//!   the addressable space is effectively unbounded (within `i32` tile coords).
//! - **Flat tops.** Each tile has an integer [`Tile::floor_height`]. Height
//!   differences between neighbours are visualised as cliffs by the default
//!   mesher; pathfinding ignores height (zones are a 2D basis).
//! - **One liquid layer.** Each tile optionally carries a [`Liquid`]
//!   (kind + depth). Multi-layer (oil-on-water) is not modelled and can be
//!   added when a game needs it.
//!
//! The liquid simulation itself (cellular flow, evaporation, etc.) is not
//! part of this module — the game's [`Simulation`](crate::Simulation) impl
//! ticks it, reading and writing [`Tile::liquid`].

use indexmap::IndexMap;
use indexmap::map as indexmap_map;

use glam::{IVec2, UVec2};

use super::grid::{Grid, SquareGrid};

/// Side length of a chunk, in tiles. A chunk holds `CHUNK_SIZE * CHUNK_SIZE`
/// tiles in row-major order.
pub const CHUNK_SIZE: u32 = 16;

const CHUNK_TILES: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Tile coordinate in the zone's local frame. Signed so the zone origin can
/// sit anywhere convenient.
pub type TileCoord = IVec2;

/// Chunk coordinate. `tile_coord.div_euclid(CHUNK_SIZE)` for each axis.
pub type ChunkCoord = IVec2;

/// Identifier for a liquid kind (water, lava, acid, …). The game owns the
/// mapping from id to material/behaviour — this module only stores the id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LiquidId(pub u32);

/// A liquid sitting on top of a tile's floor.
///
/// `depth` is measured in `floor_height` steps — the same vertical unit the
/// terrain itself is built in. `depth: 1` is one step's worth of liquid;
/// `depth: 10` fills a 10-step pit to the brim. Visible surface height is
/// `(floor_height + depth) * height_unit`, where `height_unit` is the
/// mesher's setting (not part of sim state).
///
/// Single-layer only — a tile carries one liquid kind, filled to an integer
/// number of steps. Multi-layer (oil-on-water) and sub-step precision are
/// deferred; sim games rarely need either, and either can be added later by
/// growing this encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Liquid {
    pub kind: LiquidId,
    pub depth: u8,
}

/// One cell of the terrain grid.
///
/// Defaults to a walkable tile at height 0 with no liquid — the natural
/// "empty ground" that a freshly-allocated chunk should look like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub floor_height: i32,
    pub walkable: bool,
    pub liquid: Option<Liquid>,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            floor_height: 0,
            walkable: true,
            liquid: None,
        }
    }
}

/// A `CHUNK_SIZE × CHUNK_SIZE` block of tiles, row-major (x varies fastest).
#[derive(Clone)]
pub struct Chunk {
    tiles: [Tile; CHUNK_TILES],
}

impl Chunk {
    fn new() -> Self {
        Self {
            tiles: [Tile::default(); CHUNK_TILES],
        }
    }

    pub fn tile(&self, local: UVec2) -> &Tile {
        &self.tiles[Self::index(local)]
    }

    pub fn tile_mut(&mut self, local: UVec2) -> &mut Tile {
        &mut self.tiles[Self::index(local)]
    }

    /// Row-major slice of all tiles in the chunk. `tiles()[y * CHUNK_SIZE + x]`.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    fn index(local: UVec2) -> usize {
        debug_assert!(local.x < CHUNK_SIZE && local.y < CHUNK_SIZE);
        (local.y * CHUNK_SIZE + local.x) as usize
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

/// The terrain of a single [`Zone`](super::Zone): a sparse, chunked tile grid.
///
/// Generic over [`Grid`] — the grid impl chooses the cell topology (square,
/// hex, …) while the per-cell data ([`Tile`], [`Chunk`]) stays the same.
/// Defaults to [`SquareGrid`] so unparameterized callers keep working.
pub struct Terrain<G: Grid = SquareGrid> {
    grid: G,
    chunks: IndexMap<ChunkCoord, Chunk>,
}

impl Default for Terrain<SquareGrid> {
    fn default() -> Self {
        Self::with_grid(SquareGrid)
    }
}

impl Terrain<SquareGrid> {
    /// Build an empty terrain over the default [`SquareGrid`]. For other
    /// grids, construct via [`Terrain::with_grid`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl<G: Grid> Terrain<G> {
    /// Build an empty terrain with an explicit grid instance. Use this when
    /// the grid carries configuration (none today; reserved for future
    /// orientation/scale fields).
    pub fn with_grid(grid: G) -> Self {
        Self {
            grid,
            chunks: IndexMap::new(),
        }
    }

    /// Borrow the grid impl. Meshers use this to look up corner positions
    /// and neighbour relations.
    pub fn grid(&self) -> &G {
        &self.grid
    }

    /// Read a tile. Returns `None` if the containing chunk has never been
    /// touched. Callers that want "empty ground" semantics for unallocated
    /// space can use [`Terrain::tile_or_default`].
    pub fn tile(&self, coord: TileCoord) -> Option<&Tile> {
        let (chunk_coord, local) = split_coord(coord);
        self.chunks.get(&chunk_coord).map(|c| c.tile(local))
    }

    /// Read a tile, returning [`Tile::default`] for any coordinate in an
    /// unallocated chunk. Convenient for pathfinding and other read-mostly
    /// passes that don't want to distinguish "empty ground" from "explicitly
    /// default ground".
    pub fn tile_or_default(&self, coord: TileCoord) -> Tile {
        self.tile(coord).copied().unwrap_or_default()
    }

    /// Mutable access to a tile, lazily allocating the containing chunk if
    /// needed. The new chunk is initialised with default tiles.
    pub fn tile_mut(&mut self, coord: TileCoord) -> &mut Tile {
        let (chunk_coord, local) = split_coord(coord);
        self.chunks.entry(chunk_coord).or_default().tile_mut(local)
    }

    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    /// Iterate every allocated chunk in insertion order. `IndexMap`-backed,
    /// so the order is deterministic across runs.
    pub fn chunks(&self) -> indexmap_map::Iter<'_, ChunkCoord, Chunk> {
        self.chunks.iter()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

/// Decompose a tile coordinate into its chunk coordinate and the local
/// coordinate within that chunk. Uses Euclidean division so negative tile
/// coords map to negative chunk coords with a non-negative local offset.
pub fn split_coord(coord: TileCoord) -> (ChunkCoord, UVec2) {
    let size = CHUNK_SIZE as i32;
    let chunk = IVec2::new(coord.x.div_euclid(size), coord.y.div_euclid(size));
    let local = UVec2::new(
        coord.x.rem_euclid(size) as u32,
        coord.y.rem_euclid(size) as u32,
    );
    (chunk, local)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tile_is_walkable_ground() {
        let t = Tile::default();
        assert_eq!(t.floor_height, 0);
        assert!(t.walkable);
        assert!(t.liquid.is_none());
    }

    #[test]
    fn empty_terrain_has_no_chunks() {
        let t = Terrain::new();
        assert_eq!(t.chunk_count(), 0);
        assert!(t.tile(TileCoord::new(0, 0)).is_none());
    }

    #[test]
    fn tile_or_default_synthesises_empty_ground() {
        let t = Terrain::new();
        let tile = t.tile_or_default(TileCoord::new(10_000, -10_000));
        assert_eq!(tile, Tile::default());
        assert_eq!(t.chunk_count(), 0, "read does not allocate");
    }

    #[test]
    fn tile_mut_lazily_allocates_chunk() {
        let mut t = Terrain::new();
        t.tile_mut(TileCoord::new(3, 5)).floor_height = 7;
        assert_eq!(t.chunk_count(), 1);
        assert_eq!(t.tile(TileCoord::new(3, 5)).unwrap().floor_height, 7);
    }

    #[test]
    fn writes_to_same_chunk_share_storage() {
        let mut t = Terrain::new();
        t.tile_mut(TileCoord::new(0, 0)).floor_height = 1;
        t.tile_mut(TileCoord::new(15, 15)).floor_height = 2;
        assert_eq!(t.chunk_count(), 1, "both coords in chunk (0,0)");
    }

    #[test]
    fn writes_in_different_chunks_allocate_separately() {
        let mut t = Terrain::new();
        t.tile_mut(TileCoord::new(0, 0)).floor_height = 1;
        t.tile_mut(TileCoord::new(16, 0)).floor_height = 2;
        t.tile_mut(TileCoord::new(0, 16)).floor_height = 3;
        assert_eq!(t.chunk_count(), 3);
    }

    #[test]
    fn negative_coordinates_use_euclidean_division() {
        let (chunk, local) = split_coord(TileCoord::new(-1, -1));
        assert_eq!(chunk, ChunkCoord::new(-1, -1));
        assert_eq!(local, UVec2::new(15, 15));

        let (chunk, local) = split_coord(TileCoord::new(-16, -16));
        assert_eq!(chunk, ChunkCoord::new(-1, -1));
        assert_eq!(local, UVec2::new(0, 0));

        let (chunk, local) = split_coord(TileCoord::new(-17, -17));
        assert_eq!(chunk, ChunkCoord::new(-2, -2));
        assert_eq!(local, UVec2::new(15, 15));
    }

    #[test]
    fn liquid_roundtrips() {
        let mut t = Terrain::new();
        let water = LiquidId(1);
        t.tile_mut(TileCoord::new(2, 3)).liquid = Some(Liquid {
            kind: water,
            depth: 128,
        });
        let tile = t.tile(TileCoord::new(2, 3)).unwrap();
        assert_eq!(
            tile.liquid,
            Some(Liquid {
                kind: water,
                depth: 128
            })
        );
    }

    #[test]
    fn chunk_tiles_slice_matches_row_major_layout() {
        let mut t = Terrain::new();
        t.tile_mut(TileCoord::new(0, 0)).floor_height = 10;
        t.tile_mut(TileCoord::new(1, 0)).floor_height = 20;
        t.tile_mut(TileCoord::new(0, 1)).floor_height = 30;

        let chunk = t.chunk(ChunkCoord::new(0, 0)).unwrap();
        let tiles = chunk.tiles();
        assert_eq!(tiles[0].floor_height, 10);
        assert_eq!(tiles[1].floor_height, 20);
        assert_eq!(tiles[CHUNK_SIZE as usize].floor_height, 30);
    }
}
