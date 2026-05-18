//! Streamable static-mesh asset and the glTF 2.0 loader behind it.
//!
//! ## Roles
//!
//! - [`Mesh`] owns one or more [`MeshPrimitive`]s — each a vertex buffer +
//!   index buffer pair in the canonical [`PosNormalUv`] layout, an index
//!   count, and an optional [glb material slot name](MeshPrimitive::material_name)
//!   carried straight from the source file. Plus a single [`Aabb`] covering
//!   every primitive's positions in the asset's local frame.
//! - [`decode_gltf_mesh`] parses an in-memory `.glb` byte slice and repacks
//!   every primitive of the file's single mesh into [`DecodedPrimitive`]s,
//!   applying the glTF-default Y-up → engine-native Z-up axis flip once at
//!   decode time. Pure CPU — testable without a live device.
//! - [`AssetServer::mesh`](super::AssetServer::mesh) wires those two
//!   together off the main thread, the same way the texture path does.
//!
//! ## glTF subset
//!
//! Static-mesh only (`#70`), with multi-primitive support added in `#80`.
//! The loader accepts:
//! - A single top-level mesh with one or more primitives (Blender's default
//!   "one mesh with N material slots → N primitives" shape).
//! - `POSITION`, `NORMAL`, `TEXCOORD_0` per vertex on every primitive.
//! - Indexed triangle topology.
//! - Per-primitive material slot *names* (resolved against an application
//!   [`MaterialRegistry`](super::MaterialRegistry) at draw time; parameter
//!   data on the glb material is *not* consumed — name only).
//!
//! It rejects loudly (`MeshLoadError` → `HandleError`) for:
//! - More than one mesh in the file (a Blender scene with multiple objects
//!   still has to be one-object-per-glb — out of scope for `#80`).
//! - Skins or animations declared anywhere in the file.
//! - Missing `POSITION` / `NORMAL` / `TEXCOORD_0` on any primitive.
//! - Missing indices on any primitive.
//! - Plain `.gltf` (no embedded binary blob).
//!
//! Vertex tangents, vertex colours, and PBR material parameters in the glb
//! are explicitly ignored — see #80's "out of scope" list.
//!
//! ## Y-up → Z-up axis flip
//!
//! glTF's spec-default is Y-up right-handed; currawong is Z-up right-handed
//! (CLAUDE.md: the Y/Z flip lives at the sim/view seam permanently). The
//! flip is done once at decode time on every position and normal:
//!
//! ```text
//! (x, y, z) → (x, -z, y)
//! ```
//!
//! This is rotation by +90° around the X axis. UVs are unaffected.
//!
//! ## Attribute repacking
//!
//! glTF buffer views aren't laid out in any canonical interleaved form —
//! positions, normals, and UVs typically live in three separate buffer
//! views. Rather than try to consume glTF buffers in place, we pay a
//! one-off CPU pass to repack each vertex into our 32-byte [`PosNormalUv`]
//! and upload that. Predictable downstream; the cost is bounded by
//! `n_vertices * 32 B`.

use bytemuck::cast_slice;
use glam::Vec3;
use wgpu::util::DeviceExt;

use super::vertex::PosNormalUv;
use super::visibility::Aabb;

/// GPU-resident static mesh in the canonical [`PosNormalUv`] vertex layout.
///
/// One [`Mesh`] can carry several [`MeshPrimitive`]s — Blender's default
/// "one mesh per object, one primitive per material slot" export shape.
/// Draw by iterating [`Self::primitives`], resolving each primitive's
/// [`material_name`](MeshPrimitive::material_name) against an application
/// [`MaterialRegistry`](super::MaterialRegistry), and issuing one
/// indexed draw per primitive.
///
/// Materials parameter data on the glb side is *not* consumed; the
/// contract is name-only. A primitive whose material name parses + resolves
/// in the registry draws through the registered instance; one that doesn't
/// falls back to a magenta-flavoured instance the application chose to
/// register as the fallback.
pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
    /// Axis-aligned bounding box covering every primitive's positions in
    /// the mesh's own local frame (i.e. before any per-instance world
    /// transform). Computed at decode time *after* the Y-up→Z-up flip, so
    /// it's directly comparable to engine-side bounds.
    pub bounds: Aabb,
}

impl Mesh {
    /// Upload a freshly-decoded mesh to the GPU — one vertex+index buffer
    /// pair per primitive, plus a single mesh-level [`Aabb`] covering all
    /// of them.
    ///
    /// Takes `&wgpu::Device` directly rather than a [`Renderer`](super::Renderer)
    /// so the [`AssetServer`](super::AssetServer) background loader thread
    /// can upload without reaching back into the main-thread-owned renderer.
    pub fn from_decoded_with_device(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        label: &str,
        decoded: DecodedMesh,
    ) -> Self {
        let primitives = decoded
            .primitives
            .into_iter()
            .enumerate()
            .map(|(i, p)| MeshPrimitive::upload(device, label, i, p))
            .collect();
        Self {
            primitives,
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

/// One drawable piece of a [`Mesh`] — buffers, index count, and the glb
/// material slot name the primitive declared (`None` if the glb primitive
/// had no `material` reference).
///
/// The vertex layout is always [`PosNormalUv`] and the index format is
/// always `u32`.
pub struct MeshPrimitive {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Material slot name straight from the source glb (the `name` field
    /// on the referenced glTF material). `None` if the primitive's
    /// `material` index was absent — i.e. the artist left the slot
    /// unassigned.
    ///
    /// Resolved at draw time against a [`MaterialRegistry`](super::MaterialRegistry)
    /// via [`MaterialRegistry::get_by_name`](super::MaterialRegistry::get_by_name).
    /// On a miss the caller falls back to a magenta-flavoured instance.
    pub material_name: Option<String>,
}

impl MeshPrimitive {
    fn upload(
        device: &wgpu::Device,
        mesh_label: &str,
        index: usize,
        primitive: DecodedPrimitive,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{mesh_label} primitive {index} vertices")),
            contents: cast_slice(&primitive.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{mesh_label} primitive {index} indices")),
            contents: cast_slice(&primitive.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: primitive.indices.len() as u32,
            material_name: primitive.material_name,
        }
    }
}

/// CPU-side mesh data: a `Vec` of decoded primitives plus a single overall
/// AABB. The shape [`Mesh::from_decoded_with_device`] expects.
#[derive(Clone, Debug)]
pub struct DecodedMesh {
    pub primitives: Vec<DecodedPrimitive>,
    pub bounds: Aabb,
}

/// CPU-side primitive data: [`PosNormalUv`] vertices, `u32` indices, and
/// the glb material slot name (if any).
#[derive(Clone, Debug)]
pub struct DecodedPrimitive {
    pub vertices: Vec<PosNormalUv>,
    pub indices: Vec<u32>,
    pub material_name: Option<String>,
}

/// Decode an in-memory `.glb` byte slice into a [`DecodedMesh`] in the
/// canonical [`PosNormalUv`] layout.
///
/// Accepts the multi-primitive single-mesh shape Blender exports by
/// default — one primitive per material slot — and applies the Y-up →
/// Z-up axis flip once during the position + normal repack. See the
/// module docs for the rejected-shape list.
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
    if primitive_count == 0 {
        return Err(MeshLoadError::NoPrimitives);
    }

    let mut primitives: Vec<DecodedPrimitive> = Vec::with_capacity(primitive_count);
    let mut bounds: Option<Aabb> = None;

    for primitive in mesh.primitives() {
        // The buffer-source closure ignores the buffer index: a .glb embeds
        // exactly one buffer (its BIN chunk), and `Gltf::from_slice` parks
        // it in `gltf.blob`. If the file ever references a buffer beyond
        // that (an external .bin sidecar in disguise), the reader returns
        // `None` for the missing attribute and we surface that via
        // `MissingAttribute`.
        let reader = primitive.reader(|_| Some(blob));

        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or(MeshLoadError::MissingAttribute("POSITION"))?
            .collect();
        if positions.is_empty() {
            return Err(MeshLoadError::EmptyPrimitive);
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

        // Y-up → Z-up flip applied once here on every position + normal.
        // UVs are independent of world axes. Bounds are computed *after*
        // the flip so downstream culling sees engine-native space.
        let vertices: Vec<PosNormalUv> = positions
            .iter()
            .zip(normals.iter())
            .zip(uvs.iter())
            .map(|((p, n), uv)| PosNormalUv {
                position: y_up_to_z_up(*p),
                normal: y_up_to_z_up(*n),
                uv: *uv,
            })
            .collect();

        let primitive_bounds = bounds_of_flipped_positions(&positions);
        bounds = Some(match bounds {
            Some(b) => b.union(&primitive_bounds),
            None => primitive_bounds,
        });

        let material_name = primitive
            .material()
            .name()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());

        primitives.push(DecodedPrimitive {
            vertices,
            indices,
            material_name,
        });
    }

    Ok(DecodedMesh {
        primitives,
        bounds: bounds.expect("at least one primitive, checked above"),
    })
}

/// Apply the glTF-Y-up → engine-Z-up rotation to a single 3-vector. The
/// transform is rotation by +90° around X: `(x, y, z) → (x, -z, y)`. Used
/// for both positions and normals — they share the same axis convention.
#[inline]
fn y_up_to_z_up(v: [f32; 3]) -> [f32; 3] {
    [v[0], -v[2], v[1]]
}

/// Compute bounds from a slice of *unflipped* glTF positions, applying the
/// Y-up→Z-up flip before min/max. Equivalent to flipping every position
/// first and then computing bounds, but skips the intermediate allocation.
fn bounds_of_flipped_positions(positions: &[[f32; 3]]) -> Aabb {
    let mut iter = positions.iter().map(|p| Vec3::from(y_up_to_z_up(*p)));
    let first = iter.next().expect("non-empty, checked upstream");
    let mut min = first;
    let mut max = first;
    for v in iter {
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
    /// The single mesh declared zero primitives — there's nothing to draw.
    NoPrimitives,
    /// Required vertex attribute missing on some primitive (`"POSITION"`,
    /// `"NORMAL"`, `"TEXCOORD_0"`). The engine only supports the
    /// [`PosNormalUv`](super::PosNormalUv) layout.
    MissingAttribute(&'static str),
    /// Some primitive had no `indices` accessor. Non-indexed draw is not
    /// supported — every primitive must declare indices.
    MissingIndices,
    /// An attribute's count didn't match the position count on some
    /// primitive. This is a malformed glTF, not just a feature gap — bail
    /// rather than silently stride past the end of a buffer view.
    AttributeCountMismatch {
        attribute: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Some primitive declared zero positions. Empty primitives have no
    /// useful behaviour.
    EmptyPrimitive,
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
            Self::NoPrimitives => write!(
                f,
                "glTF mesh has zero primitives; the static-mesh loader requires at least 1"
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
            Self::EmptyPrimitive => write!(f, "glTF primitive has zero positions"),
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

    /// Real Blender-authored multi-material asset (one primitive,
    /// material slot named `Lumber` — not currawong-namespaced, so
    /// downstream registry lookup will miss). Exercised here for the
    /// decoder's material-name surfacing, not for the registry binding.
    const LUMBER_BASE_GLB: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/lumber/base.glb"
    ));

    #[test]
    fn decode_unit_cube_yields_one_primitive_with_expected_counts_and_bounds() {
        let mesh = decode_gltf_mesh(UNIT_CUBE_GLB).expect("unit-cube glb must decode");
        assert_eq!(mesh.primitives.len(), 1, "cube glb has a single primitive");
        let prim = &mesh.primitives[0];
        assert_eq!(prim.vertices.len(), 24, "cube has 6 faces × 4 verts");
        assert_eq!(prim.indices.len(), 36, "cube has 6 faces × 2 tris × 3");
        // The cube is symmetric under any axis swap; bounds are the same
        // pre- and post-flip.
        assert_eq!(mesh.bounds.min, Vec3::splat(-0.5));
        assert_eq!(mesh.bounds.max, Vec3::splat(0.5));
    }

    #[test]
    fn decode_preserves_position_normal_uv_consistency() {
        let mesh = decode_gltf_mesh(UNIT_CUBE_GLB).expect("decode");
        let prim = &mesh.primitives[0];
        // Every face's four vertices share one normal — sanity that we
        // zipped position/normal/uv in lockstep rather than scrambling.
        for face_start in (0..24).step_by(4) {
            let n0 = prim.vertices[face_start].normal;
            for i in 1..4 {
                assert_eq!(
                    prim.vertices[face_start + i].normal,
                    n0,
                    "face {} verts must share a normal",
                    face_start / 4
                );
            }
        }
    }

    #[test]
    fn y_up_to_z_up_rotates_axes_correctly() {
        // +Y in glTF → +Z in engine.
        assert_eq!(y_up_to_z_up([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        // +Z in glTF → -Y in engine.
        assert_eq!(y_up_to_z_up([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0]);
        // +X is unchanged (axis we rotate around).
        assert_eq!(y_up_to_z_up([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn decode_real_blender_glb_surfaces_material_slot_name() {
        let mesh = decode_gltf_mesh(LUMBER_BASE_GLB).expect("real Blender glb must decode");
        assert!(
            !mesh.primitives.is_empty(),
            "asset must produce at least one primitive"
        );
        // base.glb has a single primitive whose material slot is named
        // "Lumber". The decoder surfaces it verbatim; downstream the
        // application registry decides whether that name resolves to a
        // registered material or falls back to magenta.
        let names: Vec<Option<&str>> = mesh
            .primitives
            .iter()
            .map(|p| p.material_name.as_deref())
            .collect();
        assert!(
            names.contains(&Some("Lumber")),
            "expected a primitive with material slot name `Lumber`, got {names:?}",
        );
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
