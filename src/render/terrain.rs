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
use glam::UVec2;

use crate::sim::{CHUNK_SIZE, ChunkCoord, LiquidId, Terrain, TileCoord};

/// One vertex of a terrain mesh. Position is in zone-local world space (Z-up,
/// matching the engine's camera convention): tile X/Y map to world X/Y,
/// `floor_height` maps to world Z. Colour is linear RGBA in `[0, 1]`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
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
/// Per-zone / per-game pluggable: swap one impl for another to change the
/// visual style without touching sim data or pathfinding.
pub trait TerrainMesher {
    fn mesh_chunk(&self, terrain: &Terrain, chunk_coord: ChunkCoord) -> ChunkMeshes;
}

// --- FlatTopsMesher -------------------------------------------------------

/// Default mesher: each tile becomes a flat quad at its `floor_height`;
/// vertical wall quads bridge to lower neighbours. Tops use [`Self::top_color`],
/// walls use [`Self::wall_color`] so cliffs read at a glance even without
/// lighting.
///
/// Liquid surfaces are flat quads at `floor_height + depth`; liquid "side
/// faces" between tiles of different liquid surface height are deferred.
pub struct FlatTopsMesher {
    /// World units per tile (XY extent of one tile's footprint).
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

    fn world_xy(&self, tx: i32, ty: i32) -> (f32, f32) {
        (tx as f32 * self.tile_size, ty as f32 * self.tile_size)
    }

    fn world_h(&self, h: i32) -> f32 {
        h as f32 * self.height_unit
    }

    fn push_quad(mesh: &mut MeshData, corners: [[f32; 3]; 4], color: [f32; 4]) {
        let base = mesh.vertices.len() as u32;
        for pos in corners {
            mesh.vertices.push(TerrainVertex { pos, color });
        }
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    fn emit_top_quad(&self, mesh: &mut MeshData, tx: i32, ty: i32, h: i32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x1 = x0 + self.tile_size;
        let y1 = y0 + self.tile_size;
        let z = self.world_h(h);
        Self::push_quad(
            mesh,
            [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
            self.top_color,
        );
    }

    /// Wall on the tile's +x face, descending from `h` to `h_low`.
    fn emit_wall_px(&self, mesh: &mut MeshData, tx: i32, ty: i32, h: i32, h_low: i32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x = x0 + self.tile_size;
        let y1 = y0 + self.tile_size;
        let zt = self.world_h(h);
        let zb = self.world_h(h_low);
        Self::push_quad(
            mesh,
            [[x, y0, zb], [x, y1, zb], [x, y1, zt], [x, y0, zt]],
            self.wall_color,
        );
    }

    fn emit_wall_nx(&self, mesh: &mut MeshData, tx: i32, ty: i32, h: i32, h_low: i32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x = x0;
        let y1 = y0 + self.tile_size;
        let zt = self.world_h(h);
        let zb = self.world_h(h_low);
        Self::push_quad(
            mesh,
            [[x, y1, zb], [x, y0, zb], [x, y0, zt], [x, y1, zt]],
            self.wall_color,
        );
    }

    /// Wall on the tile's +y face, descending from `h` to `h_low`.
    fn emit_wall_py(&self, mesh: &mut MeshData, tx: i32, ty: i32, h: i32, h_low: i32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x1 = x0 + self.tile_size;
        let y = y0 + self.tile_size;
        let zt = self.world_h(h);
        let zb = self.world_h(h_low);
        Self::push_quad(
            mesh,
            [[x1, y, zb], [x0, y, zb], [x0, y, zt], [x1, y, zt]],
            self.wall_color,
        );
    }

    fn emit_wall_ny(&self, mesh: &mut MeshData, tx: i32, ty: i32, h: i32, h_low: i32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x1 = x0 + self.tile_size;
        let y = y0;
        let zt = self.world_h(h);
        let zb = self.world_h(h_low);
        Self::push_quad(
            mesh,
            [[x0, y, zb], [x1, y, zb], [x1, y, zt], [x0, y, zt]],
            self.wall_color,
        );
    }

    fn emit_liquid_quad(&self, mesh: &mut MeshData, tx: i32, ty: i32, surface_z: f32) {
        let (x0, y0) = self.world_xy(tx, ty);
        let x1 = x0 + self.tile_size;
        let y1 = y0 + self.tile_size;
        Self::push_quad(
            mesh,
            [
                [x0, y0, surface_z],
                [x1, y0, surface_z],
                [x1, y1, surface_z],
                [x0, y1, surface_z],
            ],
            [1.0, 1.0, 1.0, 1.0],
        );
    }
}

impl TerrainMesher for FlatTopsMesher {
    fn mesh_chunk(&self, terrain: &Terrain, chunk_coord: ChunkCoord) -> ChunkMeshes {
        let mut out = ChunkMeshes::default();
        let Some(chunk) = terrain.chunk(chunk_coord) else {
            return out;
        };
        let size = CHUNK_SIZE as i32;
        let origin = chunk_coord * size;

        for ly in 0..size {
            for lx in 0..size {
                let tx = origin.x + lx;
                let ty = origin.y + ly;
                let tile = chunk.tile(UVec2::new(lx as u32, ly as u32));
                let h = tile.floor_height;

                self.emit_top_quad(&mut out.solid, tx, ty, h);

                // Walls: emit only towards a strictly lower neighbour, so the
                // higher tile owns the cliff and we never double up.
                let h_px = terrain
                    .tile_or_default(TileCoord::new(tx + 1, ty))
                    .floor_height;
                if h > h_px {
                    self.emit_wall_px(&mut out.solid, tx, ty, h, h_px);
                }
                let h_nx = terrain
                    .tile_or_default(TileCoord::new(tx - 1, ty))
                    .floor_height;
                if h > h_nx {
                    self.emit_wall_nx(&mut out.solid, tx, ty, h, h_nx);
                }
                let h_py = terrain
                    .tile_or_default(TileCoord::new(tx, ty + 1))
                    .floor_height;
                if h > h_py {
                    self.emit_wall_py(&mut out.solid, tx, ty, h, h_py);
                }
                let h_ny = terrain
                    .tile_or_default(TileCoord::new(tx, ty - 1))
                    .floor_height;
                if h > h_ny {
                    self.emit_wall_ny(&mut out.solid, tx, ty, h, h_ny);
                }

                if let Some(liq) = tile.liquid {
                    let surface = self.world_h(h + liq.depth as i32);
                    let bucket = out.liquids.entry(liq.kind).or_default();
                    self.emit_liquid_quad(bucket, tx, ty, surface);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::Liquid;

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
