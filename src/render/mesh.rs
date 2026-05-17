//! Streamable static-mesh asset and the glTF 2.0 loader behind it.
//!
//! ## Roles
//!
//! - [`Mesh`] owns a vertex buffer + index buffer pair in the canonical
//!   [`PosNormalUv`] layout, an index count, and the asset-space [`Aabb`]
//!   covering its positions. Built from a [`DecodedMesh`] via
//!   [`Mesh::from_decoded_with_device`].
//! - [`decode_gltf_mesh`] parses an in-memory `.glb` byte slice and
//!   repacks the first primitive of the first mesh into [`DecodedMesh`].
//!   Pure CPU — testable without a live device.
//! - [`AssetServer::mesh`](super::AssetServer::mesh) wires those two
//!   together off the main thread, the same way the texture path does.
//!
//! ## glTF subset
//!
//! Strict — issue #70 calls out static-mesh only. The loader rejects loudly
//! (`MeshLoadError` → `HandleError`) when asked to load any glTF that:
//!
//! - has more than one mesh, or whose single mesh has more than one
//!   primitive (we don't silently drop geometry);
//! - declares any skins or animations (asset path would need a different
//!   destination than [`Mesh`]);
//! - is missing `POSITION`, `NORMAL`, or `TEXCOORD_0` (our only vertex
//!   layout is [`PosNormalUv`]);
//! - is a plain `.gltf` (no embedded binary blob) — we read bytes through
//!   the [`Vfs`](crate::data::Vfs) and the parser, so external `.bin`
//!   sidecars would need a separate VFS path resolution we haven't
//!   committed to.
//!
//! ## Attribute repacking
//!
//! glTF buffer views aren't laid out in any canonical interleaved form —
//! positions, normals, and UVs typically live in three separate
//! buffer views, and even within one buffer view they can be strided or
//! offset oddly. Rather than try to consume glTF buffers in place, we
//! pay a one-off CPU pass to repack each vertex into our 32-byte
//! [`PosNormalUv`] and upload that. Predictable downstream; the cost is
//! bounded by `n_vertices * 32 B`.

use bytemuck::cast_slice;
use glam::Vec3;
use wgpu::util::DeviceExt;

use super::vertex::PosNormalUv;
use super::visibility::Aabb;

/// GPU-resident static mesh in the canonical [`PosNormalUv`] vertex layout.
///
/// Pair with any material that demands `PosNormalUv` (today:
/// [`PbrMaterial`](super::PbrMaterial)). Draw via
/// `pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
/// pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
/// pass.draw_indexed(0..mesh.index_count, 0, 0..n_instances)`.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Axis-aligned bounding box of the mesh's positions in its own local
    /// frame (i.e. before any per-instance world transform). Computed at
    /// decode time from the position accessor; defaulted to zero-volume at
    /// the origin for empty meshes (which the decoder rejects anyway).
    pub bounds: Aabb,
}

impl Mesh {
    /// Upload a freshly-decoded mesh to the GPU. Vertex and index buffers
    /// are created without `COPY_DST` — meshes are immutable once uploaded;
    /// re-decode + re-upload to swap content.
    ///
    /// Takes `&wgpu::Device` directly rather than a [`Renderer`](super::Renderer)
    /// so the [`AssetServer`](super::AssetServer) background loader thread
    /// can upload without reaching back into the main-thread-owned renderer.
    /// Mirrors [`Texture::from_rgba8_with_device`](super::Texture::from_rgba8_with_device).
    pub fn from_decoded_with_device(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        label: &str,
        decoded: DecodedMesh,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} vertices")),
            contents: cast_slice(&decoded.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} indices")),
            contents: cast_slice(&decoded.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: decoded.indices.len() as u32,
            bounds: decoded.bounds,
        }
    }

    /// Decode `bytes` as a `.glb` and upload the result. Convenience over
    /// [`decode_gltf_mesh`] + [`Self::from_decoded_with_device`].
    pub fn from_gltf_bytes_with_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        bytes: &[u8],
    ) -> Result<Self, MeshLoadError> {
        let decoded = decode_gltf_mesh(bytes)?;
        Ok(Self::from_decoded_with_device(
            device, queue, label, decoded,
        ))
    }
}

/// CPU-side mesh data: [`PosNormalUv`] vertices, `u32` indices, and the
/// axis-aligned bounding box of the positions. The shape
/// [`Mesh::from_decoded_with_device`] expects.
///
/// Decoders return this; tests assert on it without needing a GPU.
#[derive(Clone, Debug)]
pub struct DecodedMesh {
    pub vertices: Vec<PosNormalUv>,
    pub indices: Vec<u32>,
    pub bounds: Aabb,
}

/// Decode an in-memory `.glb` byte slice into a [`DecodedMesh`] in the
/// canonical [`PosNormalUv`] layout.
///
/// Rejects loudly for anything outside the static-mesh subset — see the
/// module docs for the exhaustive list. The intent is that the 99% of
/// exporter output that produces a single static mesh succeeds; the 1%
/// that needs skinning, multi-primitive splits, or non-`PosNormalUv`
/// attribute sets fails with a clear message rather than silently
/// producing broken geometry.
pub fn decode_gltf_mesh(bytes: &[u8]) -> Result<DecodedMesh, MeshLoadError> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(MeshLoadError::Parse)?;
    let blob = gltf
        .blob
        .as_deref()
        .ok_or(MeshLoadError::MissingBinaryBlob)?;

    if gltf.skins().len() > 0 {
        return Err(MeshLoadError::Unsupported("skins"));
    }
    if gltf.animations().len() > 0 {
        return Err(MeshLoadError::Unsupported("animations"));
    }

    let mesh_count = gltf.meshes().len();
    if mesh_count != 1 {
        return Err(MeshLoadError::WrongMeshCount(mesh_count));
    }
    let mesh = gltf.meshes().next().expect("checked count above");

    let primitive_count = mesh.primitives().len();
    if primitive_count != 1 {
        return Err(MeshLoadError::WrongPrimitiveCount(primitive_count));
    }
    let primitive = mesh.primitives().next().expect("checked count above");

    // The buffer-source closure ignores the buffer index: a .glb embeds
    // exactly one buffer (its BIN chunk), and `Gltf::from_slice` parks it
    // in `gltf.blob`. If the file ever references a buffer beyond that
    // (an external .bin sidecar in disguise), the reader returns `None`
    // for the missing attribute and we surface that via `MissingAttribute`
    // — the right shape for "we don't support this".
    let reader = primitive.reader(|_| Some(blob));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(MeshLoadError::MissingAttribute("POSITION"))?
        .collect();
    if positions.is_empty() {
        return Err(MeshLoadError::EmptyMesh);
    }

    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .ok_or(MeshLoadError::MissingAttribute("NORMAL"))?
        .collect();
    if normals.len() != positions.len() {
        return Err(MeshLoadError::AttributeCountMismatch {
            attribute: "NORMAL",
            expected: positions.len(),
            actual: normals.len(),
        });
    }

    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .ok_or(MeshLoadError::MissingAttribute("TEXCOORD_0"))?
        .into_f32()
        .collect();
    if uvs.len() != positions.len() {
        return Err(MeshLoadError::AttributeCountMismatch {
            attribute: "TEXCOORD_0",
            expected: positions.len(),
            actual: uvs.len(),
        });
    }

    let indices: Vec<u32> = reader
        .read_indices()
        .ok_or(MeshLoadError::MissingIndices)?
        .into_u32()
        .collect();

    let vertices: Vec<PosNormalUv> = positions
        .iter()
        .zip(normals.iter())
        .zip(uvs.iter())
        .map(|((p, n), uv)| PosNormalUv {
            position: *p,
            normal: *n,
            uv: *uv,
        })
        .collect();

    let bounds = compute_bounds(&positions);

    Ok(DecodedMesh {
        vertices,
        indices,
        bounds,
    })
}

/// Axis-aligned bounds of a position list. Caller guarantees non-empty
/// input; an empty slice would yield an inverted AABB and is rejected
/// upstream as [`MeshLoadError::EmptyMesh`].
fn compute_bounds(positions: &[[f32; 3]]) -> Aabb {
    let first = positions[0];
    let mut min = Vec3::from(first);
    let mut max = min;
    for p in &positions[1..] {
        let v = Vec3::from(*p);
        min = min.min(v);
        max = max.max(v);
    }
    Aabb::new(min, max)
}

/// Reasons [`decode_gltf_mesh`] rejected its input. Translates into a
/// [`HandleError`](super::HandleError) via `Display` at the asset-server
/// boundary, so the failure surfaces in the handle's `Failed` state with a
/// message the caller can act on.
#[derive(Debug)]
pub enum MeshLoadError {
    /// `gltf::Gltf::from_slice` couldn't parse the bytes.
    Parse(gltf::Error),
    /// File parsed but had no binary blob — i.e. a `.gltf` JSON without an
    /// embedded BIN chunk. Use `.glb`.
    MissingBinaryBlob,
    /// Feature outside the static-mesh subset (`"skins"`, `"animations"`).
    Unsupported(&'static str),
    /// glTF declared `count` meshes; we require exactly 1.
    WrongMeshCount(usize),
    /// The single mesh declared `count` primitives; we require exactly 1.
    WrongPrimitiveCount(usize),
    /// Required vertex attribute missing (`"POSITION"`, `"NORMAL"`,
    /// `"TEXCOORD_0"`). The engine only supports the
    /// [`PosNormalUv`](super::PosNormalUv) layout.
    MissingAttribute(&'static str),
    /// Primitive had no `indices` accessor. Non-indexed draw is not
    /// supported — every mesh must declare indices.
    MissingIndices,
    /// An attribute's count didn't match the position count. This is a
    /// malformed glTF, not just a feature gap — bail rather than silently
    /// stride past the end of a buffer view.
    AttributeCountMismatch {
        attribute: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Primitive declared zero positions. Empty meshes have no useful
    /// behaviour and would produce an inverted AABB.
    EmptyMesh,
}

impl std::fmt::Display for MeshLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "glTF parse error: {e}"),
            Self::MissingBinaryBlob => write!(
                f,
                "glTF has no embedded binary blob (pass .glb, not .gltf+.bin)"
            ),
            Self::Unsupported(feature) => write!(
                f,
                "glTF declares {feature}, which the static-mesh loader does not support"
            ),
            Self::WrongMeshCount(n) => write!(
                f,
                "glTF has {n} meshes; the static-mesh loader requires exactly 1"
            ),
            Self::WrongPrimitiveCount(n) => write!(
                f,
                "glTF mesh has {n} primitives; the static-mesh loader requires exactly 1"
            ),
            Self::MissingAttribute(name) => write!(
                f,
                "glTF primitive is missing required attribute {name} (loader expects PosNormalUv)"
            ),
            Self::MissingIndices => write!(
                f,
                "glTF primitive has no indices accessor; non-indexed draw is not supported"
            ),
            Self::AttributeCountMismatch {
                attribute,
                expected,
                actual,
            } => write!(
                f,
                "glTF primitive attribute {attribute} has {actual} entries; expected {expected} to match POSITION"
            ),
            Self::EmptyMesh => write!(f, "glTF primitive has zero positions"),
        }
    }
}

impl std::error::Error for MeshLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same checked-in `.glb` the `assets` example streams. Embedding
    /// it via `include_bytes!` means the decoder tests run against the real
    /// asset, not a divergent in-test fixture — if the generator output
    /// drifts from what the loader expects, the tests notice.
    const UNIT_CUBE_GLB: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/test/cube.glb"));

    #[test]
    fn decode_unit_cube_yields_expected_counts_and_bounds() {
        let mesh = decode_gltf_mesh(UNIT_CUBE_GLB).expect("unit-cube glb must decode");
        assert_eq!(mesh.vertices.len(), 24, "cube has 6 faces × 4 verts");
        assert_eq!(mesh.indices.len(), 36, "cube has 6 faces × 2 tris × 3");
        assert_eq!(mesh.bounds.min, Vec3::splat(-0.5));
        assert_eq!(mesh.bounds.max, Vec3::splat(0.5));
    }

    #[test]
    fn decode_preserves_position_normal_uv_consistency() {
        let mesh = decode_gltf_mesh(UNIT_CUBE_GLB).expect("decode");
        // Every face's four vertices share one normal — sanity that we
        // zipped position/normal/uv in lockstep rather than scrambling.
        for face_start in (0..24).step_by(4) {
            let n0 = mesh.vertices[face_start].normal;
            for i in 1..4 {
                assert_eq!(
                    mesh.vertices[face_start + i].normal,
                    n0,
                    "face {} verts must share a normal",
                    face_start / 4
                );
            }
        }
    }

    #[test]
    fn parse_garbage_bytes_surfaces_parse_error() {
        let err = decode_gltf_mesh(b"not a real glb").expect_err("garbage must not decode");
        assert!(
            matches!(err, MeshLoadError::Parse(_)),
            "expected Parse, got {err:?}"
        );
    }

    #[test]
    fn missing_blob_rejected_loudly() {
        // A bare .gltf JSON without the .glb wrapper has no binary blob.
        let json_only = br#"{"asset":{"version":"2.0"},"meshes":[]}"#;
        let err = decode_gltf_mesh(json_only).expect_err("no-blob input must fail");
        assert!(
            matches!(err, MeshLoadError::MissingBinaryBlob),
            "expected MissingBinaryBlob, got {err:?}"
        );
    }

    #[test]
    fn error_message_names_the_missing_attribute() {
        let err = MeshLoadError::MissingAttribute("NORMAL");
        let msg = err.to_string();
        assert!(msg.contains("NORMAL"), "error should name attribute: {msg}");
        assert!(
            msg.contains("PosNormalUv"),
            "error should name the layout: {msg}"
        );
    }

    #[test]
    fn error_message_names_the_unsupported_feature() {
        let err = MeshLoadError::Unsupported("skins");
        let msg = err.to_string();
        assert!(msg.contains("skins"), "error should name feature: {msg}");
    }
}
