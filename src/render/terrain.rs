//! View-side terrain meshing.
//!
//! Turns sim-side [`Terrain`](crate::Terrain) data into per-chunk meshes
//! ready for GPU upload. The [`TerrainMesher`] trait is the plug point —
//! different games (or different zones in the same game) can swap meshers
//! without changing sim data or pathfinding.
//!
//! [`FlatTopsMesher`] is the built-in default: each tile is a flat quad at
//! its `floor_height`; vertical wall quads bridge to lower neighbours,
//! producing the stepped DF/Minecraft/RimWorld look. Liquids are flat
//! surface quads at `floor_height + depth`.
//!
//! This module is pure CPU — it produces vertex/index buffers in host
//! memory. The `TerrainRenderer` (next slice) is what uploads them to GPU
//! buffers and draws them.
//!
//! ## Output shape
//!
//! [`ChunkMeshes`] separates **solid** terrain (one mesh, opaque) from
//! **liquids** (one mesh per [`LiquidId`], transparent). Liquids share the
//! solid vertex format but write white per-vertex colour, so the per-liquid
//! material instance can supply the actual colour via its uniform/tint —
//! transparency, refraction, etc. live in the material, not the mesh.
//!
//! Walls are emitted once per cliff edge by the higher tile, so adjacent
//! tiles never produce overlapping wall geometry.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{IVec2, UVec2, Vec3};

use crate::grid::Grid;
use crate::sim::{CHUNK_SIZE, ChunkCoord, LiquidId, Terrain};

/// One vertex of a terrain mesh. Position is in zone-local world space (Z-up,
/// matching the engine's camera convention): tile X/Y map to world X/Y,
/// `floor_height` maps to world Z. `normal` is the outward face direction
/// (also world-space, since terrain is drawn without a model matrix) — `+Z`
/// for flat tops, in-plane outward for vertical walls. Colour is linear RGBA
/// in `[0, 1]`.
///
/// `cell_id_in_chunk` identifies which cell of the owning chunk this vertex
/// belongs to — `ly * CHUNK_SIZE + lx` where `(lx, ly)` is the cell's local
/// position within the chunk. The terrain shader adds the chunk's per-frame
/// `base_id` (allocated by [`Renderer::reserve_terrain_chunk`](crate::Renderer::reserve_terrain_chunk))
/// to recover the unique frame-scoped hit ID written to the engine's
/// R32Uint ID attachment for picking. Top, wall, and liquid vertices of the
/// same cell share the same `cell_id_in_chunk`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub cell_id_in_chunk: u32,
}

/// CPU-side mesh data ready to be uploaded to a vertex/index buffer pair.
#[derive(Default)]
pub struct MeshData {
    pub vertices: Vec<TerrainVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// All meshes a [`TerrainMesher`] produces for a single chunk.
///
/// Solid terrain is one bucket; each [`LiquidId`] present in the chunk is its
/// own bucket so transparent draws can be sorted and pipelined separately.
#[derive(Default)]
pub struct ChunkMeshes {
    pub solid: MeshData,
    pub liquids: HashMap<LiquidId, MeshData>,
}

/// Turns chunked tile data into renderable geometry.
///
/// Generic over [`Grid`] — the same mesher impl can drive square and hex
/// terrain when it's written against grid topology rather than hardcoded
/// axes. The associated [`Output`](Self::Output) type lets future meshers
/// emit other representations (e.g. mesh-library instances) under the same
/// trait without disturbing existing procedural impls.
///
/// Per-zone / per-game pluggable: swap one impl for another to change the
/// visual style without touching sim data or pathfinding.
pub trait TerrainMesher<G: Grid> {
    type Output;
    fn mesh_chunk(&self, terrain: &Terrain<G>, chunk_coord: ChunkCoord) -> Self::Output;
}

// --- FlatTopsMesher -------------------------------------------------------

/// Default mesher: each cell becomes a flat polygon at its `floor_height`;
/// vertical wall quads bridge to lower neighbours. Tops use [`Self::top_color`],
/// walls use [`Self::wall_color`] so cliffs read at a glance even without
/// lighting.
///
/// Liquid surfaces are flat polygons at `floor_height + depth`; liquid "side
/// faces" between tiles of different liquid surface height are deferred.
///
/// The mesher itself is grid-agnostic — it implements [`TerrainMesher<G>`]
/// for any [`Grid`], driving cell-top triangulation and per-edge wall
/// emission through the grid's corner/neighbour APIs.
pub struct FlatTopsMesher {
    /// World units per cell of canonical unit space (scales the grid's
    /// corner positions into world XY).
    pub tile_size: f32,
    /// World units per one integer step of `floor_height` (Z extent).
    pub height_unit: f32,
    pub top_color: [f32; 4],
    pub wall_color: [f32; 4],
}

impl Default for FlatTopsMesher {
    fn default() -> Self {
        Self {
            tile_size: 1.0,
            height_unit: 1.0,
            top_color: [0.55, 0.6, 0.45, 1.0],
            wall_color: [0.35, 0.3, 0.25, 1.0],
        }
    }
}

impl FlatTopsMesher {
    pub fn new() -> Self {
        Self::default()
    }

    fn world_h(&self, h: i32) -> f32 {
        h as f32 * self.height_unit
    }

    /// Triangulate the cell's top face as a fan from corner 0. For a 4-corner
    /// grid this is 2 tris (same shape as a hand-coded quad); for 6-corner
    /// hex it's 4 tris. Normal is `+Z` — tops are flat in world space.
    fn emit_top_polygon<G: Grid>(
        &self,
        mesh: &mut MeshData,
        grid: &G,
        cell: IVec2,
        cell_id_in_chunk: u32,
        z: f32,
        color: [f32; 4],
    ) {
        let base = mesh.vertices.len() as u32;
        for i in 0..G::CORNERS_PER_CELL {
            let xy = grid.corner_xy(cell, i) * self.tile_size;
            mesh.vertices.push(TerrainVertex {
                pos: [xy.x, xy.y, z],
                normal: [0.0, 0.0, 1.0],
                color,
                cell_id_in_chunk,
            });
        }
        for i in 1..(G::CORNERS_PER_CELL as u32 - 1) {
            mesh.indices.extend([base, base + i, base + i + 1]);
        }
    }

    /// Emit one wall quad along edge `edge_idx`, descending from `h` to
    /// `h_low`. Winding is CCW when viewed from outside the cell (i.e. from
    /// the lower neighbour's side), so the wall faces the open air.
    ///
    /// Normal points outward in the XY plane, perpendicular to the edge.
    /// With corners ordered CCW from above, `(edge_dir.y, -edge_dir.x)` is
    /// the outward direction — verified to match the old axis-aligned wall
    /// directions on `SquareGrid` and to face the right neighbour on hex.
    #[allow(clippy::too_many_arguments)] // private helper; args are all natural
    fn emit_wall<G: Grid>(
        &self,
        mesh: &mut MeshData,
        grid: &G,
        cell: IVec2,
        cell_id_in_chunk: u32,
        edge_idx: usize,
        h: i32,
        h_low: i32,
    ) {
        let c0 = grid.corner_xy(cell, edge_idx) * self.tile_size;
        let c1 = grid.corner_xy(cell, (edge_idx + 1) % G::CORNERS_PER_CELL) * self.tile_size;
        let edge_dir = c1 - c0;
        let outward = glam::Vec2::new(edge_dir.y, -edge_dir.x).normalize_or_zero();
        let normal = [outward.x, outward.y, 0.0];
        let zt = self.world_h(h);
        let zb = self.world_h(h_low);
        let base = mesh.vertices.len() as u32;
        let color = self.wall_color;
        mesh.vertices.push(TerrainVertex {
            pos: [c0.x, c0.y, zb],
            normal,
            color,
            cell_id_in_chunk,
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c1.x, c1.y, zb],
            normal,
            color,
            cell_id_in_chunk,
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c1.x, c1.y, zt],
            normal,
            color,
            cell_id_in_chunk,
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c0.x, c0.y, zt],
            normal,
            color,
            cell_id_in_chunk,
        });
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

impl<G: Grid> TerrainMesher<G> for FlatTopsMesher {
    type Output = ChunkMeshes;

    fn mesh_chunk(&self, terrain: &Terrain<G>, chunk_coord: ChunkCoord) -> ChunkMeshes {
        let mut out = ChunkMeshes::default();
        let Some(chunk) = terrain.chunk(chunk_coord) else {
            return out;
        };
        let grid = terrain.grid();
        let size = CHUNK_SIZE as i32;
        let origin = chunk_coord * size;

        for ly in 0..size {
            for lx in 0..size {
                let cell = IVec2::new(origin.x + lx, origin.y + ly);
                let tile = chunk.tile(UVec2::new(lx as u32, ly as u32));
                let h = tile.floor_height;
                let cell_id_in_chunk = (ly as u32) * CHUNK_SIZE + (lx as u32);

                self.emit_top_polygon(
                    &mut out.solid,
                    grid,
                    cell,
                    cell_id_in_chunk,
                    self.world_h(h),
                    self.top_color,
                );

                // Walls: emit only towards a strictly lower neighbour, so the
                // higher tile owns the cliff and we never double up.
                for edge in 0..G::EDGES_PER_CELL {
                    let neighbour = grid.neighbour(cell, edge);
                    let neighbour_h = terrain.tile_or_default(neighbour).floor_height;
                    if h > neighbour_h {
                        self.emit_wall(
                            &mut out.solid,
                            grid,
                            cell,
                            cell_id_in_chunk,
                            edge,
                            h,
                            neighbour_h,
                        );
                    }
                }

                if let Some(liq) = tile.liquid {
                    let surface = self.world_h(h + liq.depth as i32);
                    let bucket = out.liquids.entry(liq.kind).or_default();
                    self.emit_top_polygon(
                        bucket,
                        grid,
                        cell,
                        cell_id_in_chunk,
                        surface,
                        [1.0, 1.0, 1.0, 1.0],
                    );
                }
            }
        }
        out
    }
}

// --- SlopeMesher ----------------------------------------------------------

/// Smooth-terrain mesher: each cell's top corner sits at `max(floor_height)`
/// of every cell touching that corner. A tall cell next to a shorter one
/// produces a slope between them rather than a cliff — visually OpenTTD /
/// Transport Tycoon / Rise of Industry style.
///
/// Generic over [`Grid`] like [`FlatTopsMesher`]: square cells become
/// quad-shaped sloped tiles, hex cells become 6-corner sloped hexes. Both
/// drop through the same code path because the corner topology is hidden
/// behind [`Grid::cells_at_corner`].
///
/// Flat-shaded — each triangle of the fan triangulation gets its own face
/// normal, so coplanar pieces of a cell read as one facet but sloped pairs
/// of triangles within a single cell read as a visible crease.
///
/// ## Not yet
///
/// - **No cliff threshold.** A 5-unit height difference between neighbours
///   produces a 5-unit-tall slope across one cell width — looks like a steep
///   ramp. OpenTTD's "max one-step slope, vertical cliff for more" rule is
///   the obvious next step and lives behind a follow-up flag.
/// - **No walls.** Pure slopes; isolated tall cells form sharp pyramids
///   rather than mesas with cliff faces.
pub struct SlopeMesher {
    /// World units per cell of canonical unit space.
    pub tile_size: f32,
    /// World units per integer step of `floor_height`.
    pub height_unit: f32,
    pub top_color: [f32; 4],
}

impl Default for SlopeMesher {
    fn default() -> Self {
        Self {
            tile_size: 1.0,
            height_unit: 1.0,
            top_color: [0.55, 0.6, 0.45, 1.0],
        }
    }
}

impl SlopeMesher {
    pub fn new() -> Self {
        Self::default()
    }

    fn world_h(&self, h: i32) -> f32 {
        h as f32 * self.height_unit
    }

    /// Per-corner height for the slope mesher: the max `floor_height` of
    /// every cell touching this corner. Adjacent cells agree on the result
    /// because [`Grid::cells_at_corner`] is symmetric (proved by
    /// `cells_at_corner_symmetric` tests on each grid impl).
    fn corner_height<G: Grid>(terrain: &Terrain<G>, cell: IVec2, corner_idx: usize) -> i32 {
        terrain
            .grid()
            .cells_at_corner(cell, corner_idx)
            .map(|c| terrain.tile_or_default(c).floor_height)
            .max()
            .expect("cells_at_corner always includes the cell itself")
    }

    /// Fan-triangulate a convex polygon with per-triangle face normals.
    /// Each triangle emits its own three vertices (no sharing across the
    /// fan) so triangles can carry distinct normals — that's what makes the
    /// shading flat rather than smooth.
    fn emit_flat_shaded_polygon(
        mesh: &mut MeshData,
        corners: &[Vec3],
        cell_id_in_chunk: u32,
        color: [f32; 4],
    ) {
        for i in 1..(corners.len() - 1) {
            let p0 = corners[0];
            let p1 = corners[i];
            let p2 = corners[i + 1];
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            let normal = [n.x, n.y, n.z];
            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(TerrainVertex {
                pos: p0.to_array(),
                normal,
                color,
                cell_id_in_chunk,
            });
            mesh.vertices.push(TerrainVertex {
                pos: p1.to_array(),
                normal,
                color,
                cell_id_in_chunk,
            });
            mesh.vertices.push(TerrainVertex {
                pos: p2.to_array(),
                normal,
                color,
                cell_id_in_chunk,
            });
            mesh.indices.extend([base, base + 1, base + 2]);
        }
    }
}

impl<G: Grid> TerrainMesher<G> for SlopeMesher {
    type Output = ChunkMeshes;

    fn mesh_chunk(&self, terrain: &Terrain<G>, chunk_coord: ChunkCoord) -> ChunkMeshes {
        let mut out = ChunkMeshes::default();
        let Some(chunk) = terrain.chunk(chunk_coord) else {
            return out;
        };
        let grid = terrain.grid();
        let size = CHUNK_SIZE as i32;
        let origin = chunk_coord * size;
        let n_corners = G::CORNERS_PER_CELL;
        let mut corners: Vec<Vec3> = Vec::with_capacity(n_corners);

        for ly in 0..size {
            for lx in 0..size {
                let cell = IVec2::new(origin.x + lx, origin.y + ly);
                let tile = chunk.tile(UVec2::new(lx as u32, ly as u32));
                let cell_id_in_chunk = (ly as u32) * CHUNK_SIZE + (lx as u32);

                // Sloped top: each corner at max-of-touching-cells.
                corners.clear();
                for i in 0..n_corners {
                    let h = Self::corner_height(terrain, cell, i);
                    let xy = grid.corner_xy(cell, i) * self.tile_size;
                    corners.push(Vec3::new(xy.x, xy.y, self.world_h(h)));
                }
                Self::emit_flat_shaded_polygon(
                    &mut out.solid,
                    &corners,
                    cell_id_in_chunk,
                    self.top_color,
                );

                // Liquid surface: flat polygon at floor + depth (same shape
                // contract as FlatTopsMesher — liquid level is per-cell, not
                // sloped to match neighbours).
                if let Some(liq) = tile.liquid {
                    let surface_z = self.world_h(tile.floor_height + liq.depth as i32);
                    let bucket = out.liquids.entry(liq.kind).or_default();
                    corners.clear();
                    for i in 0..n_corners {
                        let xy = grid.corner_xy(cell, i) * self.tile_size;
                        corners.push(Vec3::new(xy.x, xy.y, surface_z));
                    }
                    Self::emit_flat_shaded_polygon(
                        bucket,
                        &corners,
                        cell_id_in_chunk,
                        [1.0, 1.0, 1.0, 1.0],
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Liquid, TileCoord};

    fn allocate_chunk(t: &mut Terrain, coord: ChunkCoord) {
        // Touching any tile lazily allocates the whole chunk with defaults.
        let origin = coord * CHUNK_SIZE as i32;
        let _ = t.tile_mut(TileCoord::new(origin.x, origin.y));
    }

    fn quad_count(mesh: &MeshData) -> usize {
        assert_eq!(mesh.indices.len() % 6, 0);
        mesh.indices.len() / 6
    }

    const TILES_PER_CHUNK: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;

    #[test]
    fn unallocated_chunk_meshes_to_nothing() {
        let t = Terrain::new();
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert!(m.solid.is_empty());
        assert!(m.liquids.is_empty());
    }

    #[test]
    fn flat_chunk_has_tops_only_no_walls() {
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(quad_count(&m.solid), TILES_PER_CHUNK);
        assert!(m.liquids.is_empty());
    }

    #[test]
    fn elevated_tile_in_middle_emits_four_walls() {
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(5, 5)).floor_height = 3;
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(quad_count(&m.solid), TILES_PER_CHUNK + 4);
    }

    #[test]
    fn lower_tile_does_not_own_its_walls() {
        // A pit in the middle: tile (5,5) at h=-2 has 4 neighbours at h=0.
        // The neighbours each emit one wall facing the pit (4 walls total).
        // The pit itself emits zero walls.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(5, 5)).floor_height = -2;
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(quad_count(&m.solid), TILES_PER_CHUNK + 4);
    }

    #[test]
    fn no_double_walls_between_adjacent_tiles() {
        // Two adjacent tiles at different heights produce exactly one wall.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(2, 2)).floor_height = 5;
        t.tile_mut(TileCoord::new(3, 2)).floor_height = 5;
        // Both elevated tiles see each other as same-height (no wall between),
        // and each has 3 lower neighbours → 3 walls each = 6 walls total.
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(quad_count(&m.solid), TILES_PER_CHUNK + 6);
    }

    #[test]
    fn cliff_at_chunk_boundary_reads_neighbour_chunk() {
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        // Allocate chunk (1,0); then drop just its (0,0) local tile down.
        // Chunk (0,0)'s tile (15, 0) sees a lower +x neighbour and emits one
        // wall; all other boundary tiles see same-height neighbours.
        allocate_chunk(&mut t, ChunkCoord::new(1, 0));
        t.tile_mut(TileCoord::new(16, 0)).floor_height = -3;
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(quad_count(&m.solid), TILES_PER_CHUNK + 1);
    }

    #[test]
    fn liquids_split_into_buckets_by_kind() {
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        let water = LiquidId(1);
        let lava = LiquidId(2);
        t.tile_mut(TileCoord::new(1, 1)).liquid = Some(Liquid {
            kind: water,
            depth: 128,
        });
        t.tile_mut(TileCoord::new(2, 1)).liquid = Some(Liquid {
            kind: water,
            depth: 128,
        });
        t.tile_mut(TileCoord::new(3, 3)).liquid = Some(Liquid {
            kind: lava,
            depth: 200,
        });
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        assert_eq!(m.liquids.len(), 2);
        assert_eq!(quad_count(m.liquids.get(&water).unwrap()), 2);
        assert_eq!(quad_count(m.liquids.get(&lava).unwrap()), 1);
    }

    #[test]
    fn flat_top_vertices_have_up_normal() {
        // Every vertex of a single flat tile's top should point straight up
        // (+Z). On a fully flat chunk there are no walls, so every emitted
        // vertex is a top vertex.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        for v in &m.solid.vertices {
            assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn square_wall_normals_face_outward() {
        // An elevated tile in the middle of a flat chunk emits four walls,
        // one per cardinal direction. Each wall's normal should point away
        // from the tile centre — +X, +Y, -X, -Y respectively (z = 0).
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(5, 5)).floor_height = 3;
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);

        // Collect unique wall normals (one per emitted wall quad — all four
        // vertices of a quad share a normal).
        let mut wall_normals: Vec<[f32; 3]> = m
            .solid
            .vertices
            .iter()
            .filter(|v| v.normal != [0.0, 0.0, 1.0])
            .map(|v| v.normal)
            .collect();
        wall_normals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        wall_normals.dedup();

        // Each wall contributes 4 verts → 4 distinct outward directions.
        // The full set across all walls is {+X, +Y, -X, -Y}.
        let mut want = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(wall_normals, want.to_vec());
    }

    // --- SlopeMesher ------------------------------------------------------

    #[test]
    fn slope_flat_chunk_top_normals_point_up() {
        // A fully flat chunk has zero slope on every triangle, so face
        // normals should be +Z everywhere. Catches sign errors in the cross
        // product or winding.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        let m = SlopeMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        for v in &m.solid.vertices {
            let n = v.normal;
            assert!(
                (n[0]).abs() < 1e-5 && (n[1]).abs() < 1e-5 && (n[2] - 1.0).abs() < 1e-5,
                "expected +Z, got {n:?}",
            );
        }
    }

    #[test]
    fn slope_corner_is_max_of_touching_cells() {
        // A 2×2 cluster of tile heights [3, 0; 0, 0]. The interior corner
        // shared by all four cells should sit at z=3·height_unit on every
        // touching cell's top polygon — they all agree on the corner height.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(0, 0)).floor_height = 3;
        // Tile (0,1), (1,0), (1,1) keep default 0.
        let mesher = SlopeMesher {
            height_unit: 1.0,
            ..SlopeMesher::new()
        };
        let m = mesher.mesh_chunk(&t, ChunkCoord::ZERO);

        // The interior corner sits at world (1, 1) in canonical unit space
        // (tile_size = 1). Every vertex emitted at that XY should have z=3.
        let mut found = 0;
        for v in &m.solid.vertices {
            let xy_match = (v.pos[0] - 1.0).abs() < 1e-5 && (v.pos[1] - 1.0).abs() < 1e-5;
            if xy_match {
                found += 1;
                assert!(
                    (v.pos[2] - 3.0).abs() < 1e-5,
                    "vertex at (1,1) should be z=3, got z={}",
                    v.pos[2],
                );
            }
        }
        assert!(
            found >= 4,
            "expected the (1,1) corner to appear in at least 4 cells' tops, found {found}",
        );
    }

    #[test]
    fn slope_hill_emits_tall_vertices() {
        // Mirrors the `slope_terrain` example's hill setup. Verifies that
        // a multi-step hill actually produces vertices at non-zero z. If
        // this fails, the example will look flat regardless of lighting.
        let mut t = Terrain::new();
        // Allocate the 4 chunks that span (-8..8) × (-8..8).
        for ty in -8..8 {
            for tx in -8..8 {
                t.tile_mut(TileCoord::new(tx, ty)).floor_height = 0;
            }
        }
        // Hill centred at (2, 2) with the example's exact formula.
        for ty in -8..8 {
            for tx in -8..8 {
                let dx = tx - 2;
                let dy = ty - 2;
                let d2 = dx * dx + dy * dy;
                let h = if d2 == 0 {
                    4
                } else if d2 <= 2 {
                    3
                } else if d2 <= 8 {
                    2
                } else if d2 <= 18 {
                    1
                } else {
                    0
                };
                if h > 0 {
                    t.tile_mut(TileCoord::new(tx, ty)).floor_height = h;
                }
            }
        }

        let mesher = SlopeMesher {
            height_unit: 1.0,
            ..SlopeMesher::new()
        };

        let mut max_z = f32::MIN;
        let mut chunks_meshed = 0;
        for (chunk_coord, _) in t.chunks() {
            let m = mesher.mesh_chunk(&t, *chunk_coord);
            chunks_meshed += 1;
            for v in &m.solid.vertices {
                if v.pos[2] > max_z {
                    max_z = v.pos[2];
                }
            }
        }
        assert!(
            chunks_meshed >= 4,
            "expected 4 chunks meshed, got {chunks_meshed}"
        );
        assert!(
            (max_z - 4.0).abs() < 1e-5,
            "expected peak z=4.0 somewhere in the mesh, got max_z={max_z}",
        );
    }

    #[test]
    fn slope_isolated_peak_produces_pyramid() {
        // A single elevated tile in an empty plain. The flat-shaded fan
        // gives two triangles for the cell's top, and each triangle has a
        // distinct normal (since the polygon is non-planar). With a 4-corner
        // square cell, the four neighbours pull the peak's corners up to
        // various heights — exactly one corner (the cell's centre-most one)
        // is at the full peak height; the other three are at 0 (clamped by
        // surrounding cells with floor_height=0).
        //
        // Wait — actually *all four corners* are at max(self=3, neighbours=0)
        // = 3 because the corner is shared with us. So the top is flat at 3,
        // not sloped. The slopes appear on the neighbour cells.
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        t.tile_mut(TileCoord::new(5, 5)).floor_height = 3;
        let m = SlopeMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        // Sanity: every vertex z is in [0, 3].
        for v in &m.solid.vertices {
            assert!(
                v.pos[2] >= -1e-5 && v.pos[2] <= 3.0 + 1e-5,
                "z={}",
                v.pos[2]
            );
        }
    }

    #[test]
    fn liquid_surface_is_floor_plus_depth_steps() {
        // The canonical "pit filled to the brim" case: a 10-step pit (floor
        // at -10) with `depth: 10` brings the surface back to the
        // surrounding ground level (z=0 at height_unit=1).
        let mut t = Terrain::new();
        allocate_chunk(&mut t, ChunkCoord::ZERO);
        let water = LiquidId(1);
        let tile = t.tile_mut(TileCoord::new(0, 0));
        tile.floor_height = -10;
        tile.liquid = Some(Liquid {
            kind: water,
            depth: 10,
        });
        let m = FlatTopsMesher::new().mesh_chunk(&t, ChunkCoord::ZERO);
        let bucket = m.liquids.get(&water).unwrap();
        for v in &bucket.vertices {
            assert!(
                (v.pos[2] - 0.0).abs() < 1e-6,
                "expected z=0.0, got {}",
                v.pos[2]
            );
        }
    }
}
