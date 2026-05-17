//! One-shot generator for `assets/test/cube.glb` — the unit-cube glTF the
//! `assets` example streams and the mesh decoder's tests `include_bytes!`.
//!
//! Run once when bootstrapping or after the encoder logic changes:
//!
//! ```text
//! cargo run --bin gen_test_assets
//! ```
//!
//! Commits the resulting `.glb` alongside the source. The mesh-decoder
//! tests read it via `include_bytes!`, so the same file is the single
//! source of truth for the streamed example asset and the loader's test
//! fixture — if the generator drifts from what the loader expects, the
//! tests notice.
//!
//! ## What it emits
//!
//! A minimal `.glb` (glTF Binary) packaging a unit cube centred at the
//! origin (`[-0.5, 0.5]^3`) with per-face flat normals, UVs spanning 0..1
//! on each face, and `u16` indices. Identical topology to
//! [`currawong::PrimitiveMesh::cube(Vec3::ONE)`] so the engine's existing
//! cube generator stays the canonical reference for "what a cube looks
//! like" — the glb is just a glTF wrapping of the same geometry.

use std::path::Path;

use currawong::PrimitiveMesh;
use currawong::glam::Vec3;

/// `gltf` accessor `componentType` for `f32`. The glTF 2.0 spec encodes
/// these as the OpenGL GL_FLOAT etc. integer constants.
const COMPONENT_TYPE_FLOAT: u32 = 5126;
/// `gltf` accessor `componentType` for `u16`.
const COMPONENT_TYPE_U16: u32 = 5123;
/// `gltf` primitive `mode` for triangle list.
const MODE_TRIANGLES: u32 = 4;

fn main() {
    let mesh = PrimitiveMesh::cube(Vec3::ONE);
    assert_eq!(mesh.vertices.len(), 24, "PrimitiveMesh::cube layout drift");
    assert_eq!(mesh.indices.len(), 36, "PrimitiveMesh::cube layout drift");

    let glb = encode_unit_cube_glb(&mesh);

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/test");
    std::fs::create_dir_all(&out_dir).expect("create assets/test");
    let out_path = out_dir.join("cube.glb");
    std::fs::write(&out_path, &glb).expect("write cube.glb");

    println!("wrote {} ({} bytes)", out_path.display(), glb.len());
}

/// Encode a single-primitive glTF Binary file for the given mesh.
///
/// Layout: positions, then normals, then UVs, then indices — four
/// contiguous buffer views into one buffer. Tight-packed, no interleaving,
/// no padding inside the buffer (each accessor's element size happens to
/// be 4-byte-aligned naturally). The two .glb chunks (JSON and BIN) are
/// padded out to a 4-byte boundary as the spec requires.
fn encode_unit_cube_glb(mesh: &PrimitiveMesh) -> Vec<u8> {
    // Positions: 24 × vec3<f32> = 288 B.
    let positions_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.position.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let positions_len = positions_bytes.len();

    // Normals: 24 × vec3<f32> = 288 B.
    let normals_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.normal.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let normals_len = normals_bytes.len();

    // UVs: 24 × vec2<f32> = 192 B.
    let uvs_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.uv.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let uvs_len = uvs_bytes.len();

    // Indices: 36 × u16 = 72 B. `PrimitiveMesh::cube` produces a u32 index
    // buffer but every value is < 24, so it round-trips through u16 safely
    // — and a smaller index buffer is more honest about a 24-vertex mesh.
    let indices_bytes: Vec<u8> = mesh
        .indices
        .iter()
        .flat_map(|&i| {
            let v: u16 = i.try_into().expect("cube indices fit in u16");
            v.to_le_bytes()
        })
        .collect();
    let indices_len = indices_bytes.len();

    let positions_offset = 0;
    let normals_offset = positions_offset + positions_len;
    let uvs_offset = normals_offset + normals_len;
    let indices_offset = uvs_offset + uvs_len;
    let buffer_len = indices_offset + indices_len;

    // Bounds of the position list — required on the POSITION accessor per
    // the glTF 2.0 spec. For our unit cube this is just ±0.5.
    let half = Vec3::ONE * 0.5;
    let pos_min = format!("[{:.6},{:.6},{:.6}]", -half.x, -half.y, -half.z);
    let pos_max = format!("[{:.6},{:.6},{:.6}]", half.x, half.y, half.z);

    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"currawong gen_test_assets"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"cube"}}],"meshes":[{{"name":"cube","primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"mode":{mode}}}]}}],"accessors":[{{"bufferView":0,"componentType":{ct_float},"count":{vcount},"type":"VEC3","min":{pos_min},"max":{pos_max}}},{{"bufferView":1,"componentType":{ct_float},"count":{vcount},"type":"VEC3"}},{{"bufferView":2,"componentType":{ct_float},"count":{vcount},"type":"VEC2"}},{{"bufferView":3,"componentType":{ct_u16},"count":{icount},"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":{p_off},"byteLength":{p_len}}},{{"buffer":0,"byteOffset":{n_off},"byteLength":{n_len}}},{{"buffer":0,"byteOffset":{u_off},"byteLength":{u_len}}},{{"buffer":0,"byteOffset":{i_off},"byteLength":{i_len}}}],"buffers":[{{"byteLength":{buf_len}}}]}}"#,
        mode = MODE_TRIANGLES,
        ct_float = COMPONENT_TYPE_FLOAT,
        ct_u16 = COMPONENT_TYPE_U16,
        vcount = mesh.vertices.len(),
        icount = mesh.indices.len(),
        pos_min = pos_min,
        pos_max = pos_max,
        p_off = positions_offset,
        p_len = positions_len,
        n_off = normals_offset,
        n_len = normals_len,
        u_off = uvs_offset,
        u_len = uvs_len,
        i_off = indices_offset,
        i_len = indices_len,
        buf_len = buffer_len,
    );

    // glb chunks must be padded to 4-byte boundaries — JSON with ASCII
    // space (0x20), BIN with zero. The padding is part of the chunk's
    // own length field.
    let mut json_bytes = json.into_bytes();
    let json_pad = (4 - json_bytes.len() % 4) % 4;
    json_bytes.extend(std::iter::repeat_n(b' ', json_pad));

    let mut bin_bytes = Vec::with_capacity(buffer_len);
    bin_bytes.extend_from_slice(&positions_bytes);
    bin_bytes.extend_from_slice(&normals_bytes);
    bin_bytes.extend_from_slice(&uvs_bytes);
    bin_bytes.extend_from_slice(&indices_bytes);
    let bin_pad = (4 - bin_bytes.len() % 4) % 4;
    bin_bytes.extend(std::iter::repeat_n(0u8, bin_pad));

    // glb header (12 B) + JSON chunk header (8 B) + JSON + BIN chunk
    // header (8 B) + BIN.
    let total_len: u32 = (12 + 8 + json_bytes.len() + 8 + bin_bytes.len()) as u32;

    let mut out = Vec::with_capacity(total_len as usize);
    // Header: magic, version, total length.
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total_len.to_le_bytes());
    // JSON chunk: length, type (`JSON` ASCII), payload.
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    // BIN chunk: length, type (`BIN\0`), payload.
    out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin_bytes);

    assert_eq!(
        out.len() as u32,
        total_len,
        "header length must match actual byte count"
    );
    out
}
