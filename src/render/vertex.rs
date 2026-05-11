//! Canonical vertex layouts.
//!
//! The engine ships a small *closed* set of vertex layouts that materials
//! pick from — same architectural move as
//! [`SlotKind`](super::SlotKind): typed end-to-end, no string keys, no
//! dynamic attribute negotiation. A material that wants normal-mapped PBR
//! statically demands [`PosNormalUv`] (or [`PosNormalUvTangent`] when it
//! exists); meshes are built for one layout and either match or don't.
//!
//! Adding a new layout is a deliberate code change: add the `Pod` struct,
//! its `ATTRIBS` const, and (typically) one matching material pipeline.

use bytemuck::{Pod, Zeroable};

/// Per-vertex position + normal + uv. The minimum a single-light-shaded
/// material needs. 32 bytes, tight-packed, no tangents.
///
/// ```text
/// offset  0   12  24  end=32
///         pos nrm uv
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct PosNormalUv {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl PosNormalUv {
    /// `VertexAttribute` array for the three fields at shader locations
    /// `start_location`, `start_location + 1`, `start_location + 2`.
    /// Materials use this directly in their `VertexBufferLayout`.
    pub const fn attributes(start_location: u32) -> [wgpu::VertexAttribute; 3] {
        [
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: start_location,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: start_location + 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: start_location + 2,
                format: wgpu::VertexFormat::Float32x2,
            },
        ]
    }

    /// Stride in bytes.
    pub const STRIDE: u64 = 32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_matches_pod_size() {
        // If this fails, the const STRIDE is out of sync with the Pod
        // layout — pipelines that use it will read garbage.
        assert_eq!(
            PosNormalUv::STRIDE,
            std::mem::size_of::<PosNormalUv>() as u64
        );
    }
}
