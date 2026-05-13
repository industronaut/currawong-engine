//! GPU-side terrain state: per-chunk vertex/index buffers plus the draw
//! routine that consumes them.
//!
//! The [`TerrainRenderer`] doesn't own the mesher or the material — both
//! are passed in at the call site. This keeps it pluggable: swap meshers,
//! share one renderer across views, etc.
//!
//! Mutation strategy is "rebuild the affected chunk" — call
//! [`TerrainRenderer::rebuild_chunk`] whenever a chunk's tiles change.
//! Reuse of GPU buffers across rebuilds is deferred; edits today reallocate
//! the chunk's buffers, which is fine while edit cadence is low.

use std::collections::HashMap;

use crate::sim::{ChunkCoord, Grid, LiquidId, Terrain};

use super::renderer::Renderer;
use super::terrain::{ChunkMeshes, MeshData, TerrainMesher, TerrainVertex};
use super::terrain_material::TerrainMaterialInstance;

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn upload(renderer: &Renderer, data: &MeshData) -> Self {
        let vertices = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain chunk vertices"),
            size: (data.vertices.len() * std::mem::size_of::<TerrainVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&data.vertices));

        let indices = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain chunk indices"),
            size: (data.indices.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        renderer
            .queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&data.indices));

        Self {
            vertices,
            indices,
            index_count: data.indices.len() as u32,
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

struct ChunkBuffers {
    solid: Option<GpuMesh>,
    liquids: HashMap<LiquidId, GpuMesh>,
}

impl ChunkBuffers {
    fn build(renderer: &Renderer, meshes: ChunkMeshes) -> Self {
        let solid = (!meshes.solid.is_empty()).then(|| GpuMesh::upload(renderer, &meshes.solid));
        let liquids = meshes
            .liquids
            .into_iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|(id, m)| (id, GpuMesh::upload(renderer, &m)))
            .collect();
        Self { solid, liquids }
    }
}

/// Per-chunk GPU mesh cache for terrain rendering.
///
/// Sync with sim state by calling [`Self::rebuild_chunk`] when a chunk's
/// tiles change, or [`Self::rebuild_all`] for a one-shot rebuild of every
/// allocated chunk in the [`Terrain`].
#[derive(Default)]
pub struct TerrainRenderer {
    chunks: HashMap<ChunkCoord, ChunkBuffers>,
}

impl TerrainRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Mesh `chunk` with `mesher` and replace the cached GPU buffers.
    pub fn rebuild_chunk<G: Grid>(
        &mut self,
        renderer: &Renderer,
        terrain: &Terrain<G>,
        mesher: &dyn TerrainMesher<G, Output = ChunkMeshes>,
        chunk: ChunkCoord,
    ) {
        let meshes = mesher.mesh_chunk(terrain, chunk);
        self.chunks
            .insert(chunk, ChunkBuffers::build(renderer, meshes));
    }

    /// Mesh every allocated chunk in `terrain` from scratch, discarding any
    /// previously cached buffers. Useful for first-frame setup or after a
    /// wholesale terrain reload.
    pub fn rebuild_all<G: Grid>(
        &mut self,
        renderer: &Renderer,
        terrain: &Terrain<G>,
        mesher: &dyn TerrainMesher<G, Output = ChunkMeshes>,
    ) {
        self.chunks.clear();
        let coords: Vec<ChunkCoord> = terrain.chunks().map(|(c, _)| *c).collect();
        for c in coords {
            self.rebuild_chunk(renderer, terrain, mesher, c);
        }
    }

    /// Record opaque solid-terrain draws. Caller must have already bound
    /// [`TerrainMaterial::opaque_pipeline`](super::TerrainMaterial::opaque_pipeline)
    /// and the camera bind group at index 0.
    pub fn draw_solid(&self, pass: &mut wgpu::RenderPass<'_>, solid: &TerrainMaterialInstance) {
        pass.set_bind_group(1, solid.bind_group(), &[]);
        for buffers in self.chunks.values() {
            if let Some(mesh) = &buffers.solid {
                mesh.draw(pass);
            }
        }
    }

    /// Record transparent liquid draws keyed by [`LiquidId`]. Caller must
    /// have already bound
    /// [`TerrainMaterial::transparent_pipeline`](super::TerrainMaterial::transparent_pipeline)
    /// and the camera bind group at index 0. Any liquid kind present in the
    /// chunks but missing from `instances` is silently skipped.
    pub fn draw_liquids(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        instances: &HashMap<LiquidId, TerrainMaterialInstance>,
    ) {
        for (id, instance) in instances {
            pass.set_bind_group(1, instance.bind_group(), &[]);
            for buffers in self.chunks.values() {
                if let Some(mesh) = buffers.liquids.get(id) {
                    mesh.draw(pass);
                }
            }
        }
    }
}
