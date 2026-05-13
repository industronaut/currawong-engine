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
use glam::{IVec2, UVec2};

use crate::sim::{CHUNK_SIZE, ChunkCoord, Grid, LiquidId, Terrain};

/// One vertex of a terrain mesh. Position is in zone-local world space (Z-up,
/// matching the engine's camera convention): tile X/Y map to world X/Y,
/// `floor_height` maps to world Z. `normal` is the outward face direction
/// (also world-space, since terrain is drawn without a model matrix) — `+Z`
/// for flat tops, in-plane outward for vertical walls. Colour is linear RGBA
/// in `[0, 1]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
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
    fn emit_wall<G: Grid>(
        &self,
        mesh: &mut MeshData,
        grid: &G,
        cell: IVec2,
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
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c1.x, c1.y, zb],
            normal,
            color,
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c1.x, c1.y, zt],
            normal,
            color,
        });
        mesh.vertices.push(TerrainVertex {
            pos: [c0.x, c0.y, zt],
            normal,
            color,
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

                self.emit_top_polygon(&mut out.solid, grid, cell, self.world_h(h), self.top_color);

                // Walls: emit only towards a strictly lower neighbour, so the
                // higher tile owns the cliff and we never double up.
                for edge in 0..G::EDGES_PER_CELL {
                    let neighbour = grid.neighbour(cell, edge);
                    let neighbour_h = terrain.tile_or_default(neighbour).floor_height;
                    if h > neighbour_h {
                        self.emit_wall(&mut out.solid, grid, cell, edge, h, neighbour_h);
                    }
                }

                if let Some(liq) = tile.liquid {
                    let surface = self.world_h(h + liq.depth as i32);
                    let bucket = out.liquids.entry(liq.kind).or_default();
                    self.emit_top_polygon(bucket, grid, cell, surface, [1.0, 1.0, 1.0, 1.0]);
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
