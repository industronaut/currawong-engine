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
//!
//! Per chunk we also hold a 16-byte uniform buffer + bind group for the
//! per-frame hit-ID base. [`Self::draw_solid`] reserves a base ID via
//! [`Renderer::reserve_terrain_chunk`](super::Renderer::reserve_terrain_chunk)
//! once per chunk per frame and writes it into the chunk's uniform; the
//! terrain shader sums it with each vertex's chunk-local cell index to
//! produce the unique frame-scoped ID written to the engine's R32Uint
//! attachment for picking (#56 PR 2). Liquids draw through the transparent
//! pipeline that opts out of the ID slot, so [`Self::draw_liquids`] doesn't
//! need the renderer.

use std::collections::HashMap;

use crate::grid::Grid;
use crate::sim::{ChunkCoord, LiquidId, Terrain, ZoneId};

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

/// Per-chunk uniform buffer + bind group for the hit-ID base. Bound at
/// `@group(3)` by `draw_solid`; the engine's `id_base_layout` is the
/// source of truth for the bind-group shape.
struct ChunkIdBinding {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ChunkIdBinding {
    fn new(renderer: &Renderer) -> Self {
        // 16 bytes — std140 minimum uniform-buffer size; the WGSL struct
        // pads its single u32 out to 16 bytes for the same reason.
        let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain chunk id-base uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = renderer
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Terrain chunk id-base bg"),
                layout: renderer.id_base_layout(),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
        Self { buffer, bind_group }
    }

    fn write(&self, renderer: &Renderer, base_id: u32) {
        // 16-byte payload: base_id at offset 0, three padding u32s after.
        let payload: [u32; 4] = [base_id, 0, 0, 0];
        renderer
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&payload));
    }
}

struct ChunkBuffers {
    solid: Option<GpuMesh>,
    liquids: HashMap<LiquidId, GpuMesh>,
    id_binding: ChunkIdBinding,
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
        Self {
            solid,
            liquids,
            id_binding: ChunkIdBinding::new(renderer),
        }
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

    /// Record opaque solid-terrain draws. Caller must have already bound:
    /// - [`TerrainMaterial::opaque_pipeline`](super::TerrainMaterial::opaque_pipeline)
    /// - camera bind group at slot 0
    /// - scene environment bind group at slot 1
    ///   ([`Renderer::scene_bind_group`](super::Renderer::scene_bind_group))
    ///
    /// Per chunk, this reserves a fresh hit-ID base from the engine's
    /// frame ID table, writes it into the chunk's uniform, binds the chunk's
    /// id-base bind group at slot 3, then issues the draw. `zone` identifies
    /// which zone these chunks belong to so the eventual readback resolves
    /// to the right [`HitTarget`](super::HitTarget).
    pub fn draw_solid(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        renderer: &Renderer,
        zone: ZoneId,
        solid: &TerrainMaterialInstance,
    ) {
        pass.set_bind_group(2, solid.bind_group(), &[]);
        for (coord, buffers) in &self.chunks {
            let Some(mesh) = &buffers.solid else { continue };
            let base_id = renderer.reserve_terrain_chunk(zone, *coord);
            buffers.id_binding.write(renderer, base_id);
            pass.set_bind_group(3, &buffers.id_binding.bind_group, &[]);
            mesh.draw(pass);
        }
    }

    /// Record transparent liquid draws keyed by [`LiquidId`]. Caller must
    /// have already bound:
    /// - [`TerrainMaterial::transparent_pipeline`](super::TerrainMaterial::transparent_pipeline)
    /// - camera bind group at slot 0
    /// - scene environment bind group at slot 1
    ///
    /// Liquids opt out of the hit-ID attachment, so this binds the chunk's
    /// id-base bind group at slot 3 only to satisfy the pipeline layout —
    /// the value never reaches the ID buffer.
    ///
    /// Any liquid kind present in the chunks but missing from `instances` is
    /// silently skipped.
    pub fn draw_liquids(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        instances: &HashMap<LiquidId, TerrainMaterialInstance>,
    ) {
        for (id, instance) in instances {
            pass.set_bind_group(2, instance.bind_group(), &[]);
            for buffers in self.chunks.values() {
                if let Some(mesh) = buffers.liquids.get(id) {
                    pass.set_bind_group(3, &buffers.id_binding.bind_group, &[]);
                    mesh.draw(pass);
                }
            }
        }
    }
}
