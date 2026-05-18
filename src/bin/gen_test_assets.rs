//! One-shot generator for the engine's committed placeholder assets.
//!
//! Run this once when bootstrapping or after the encoder logic changes:
//!
//! ```text
//! cargo run --bin gen_test_assets
//! ```
//!
//! All output files are committed to the repo alongside the source. There
//! is no runtime asset-pipeline yet — examples just load these files via
//! [`AssetServer`](currawong::AssetServer) at runtime.
//!
//! ## What it emits
//!
//! Two flavours of asset:
//!
//! - **`assets/test/cube.glb`** — the unit cube the [`assets`](../examples/assets.rs)
//!   example streams, and the mesh decoder's tests `include_bytes!` against.
//!   Locked-in topology: matches [`PrimitiveMesh::cube(Vec3::ONE)`]
//!   exactly, with `u16` indices. Don't touch this without also touching the
//!   decoder fixture expectations.
//! - **`assets/models/**` + `assets/textures/**`** — the lumber-camp
//!   placeholder glTFs and PNGs the lumber-camp example streams (oak +
//!   pine trees, the camp building, the lumberjack). Origin-at-base meshes
//!   in [`PosNormalUv`] layout, `u32` indices; tiny solid-colour PNGs with
//!   a touch of luminance noise so the texture sample is visibly bound
//!   (rather than a flat constant that could be confused with the
//!   `albedo_factor`).
//!
//! Total committed size is a few tens of KB across all files combined —
//! the issue cap was "single-digit MB", and we're well under.

use std::path::{Path, PathBuf};

use currawong::PrimitiveMesh;
use currawong::glam::Vec3;
use image::{ImageBuffer, Rgba};

/// `gltf` accessor `componentType` for `f32`. The glTF 2.0 spec encodes
/// these as the OpenGL GL_FLOAT etc. integer constants.
const COMPONENT_TYPE_FLOAT: u32 = 5126;
/// `gltf` accessor `componentType` for `u16`.
const COMPONENT_TYPE_U16: u32 = 5123;
/// `gltf` accessor `componentType` for `u32`. Used by the lumber-camp
/// placeholders, whose vertex counts overflow u16.
const COMPONENT_TYPE_U32: u32 = 5125;
/// `gltf` primitive `mode` for triangle list.
const MODE_TRIANGLES: u32 = 4;

fn main() {
    // --- assets/test/cube.glb (legacy fixture, u16 indices) --------------
    let cube = PrimitiveMesh::cube(Vec3::ONE);
    assert_eq!(cube.vertices.len(), 24, "PrimitiveMesh::cube layout drift");
    assert_eq!(cube.indices.len(), 36, "PrimitiveMesh::cube layout drift");
    let cube_path = assets_path(&["test", "cube.glb"]);
    write_bytes(&cube_path, &encode_unit_cube_glb(&cube));

    // --- assets/models/** (lumber-camp placeholder meshes) ----------------
    //
    // Origin-at-base for each kind so the sim can plant objects at Z = 0
    // and the mesh's footprint sits flat on the ground without any
    // sim-side height offset. The view-side template's `visual_bounds`
    // declared in code mirrors what these meshes occupy.

    let oak = ground_origin(PrimitiveMesh::cone(0.60, 2.0, 16, true), 1.0);
    write_bytes(
        &assets_path(&["models", "trees", "oak.glb"]),
        &encode_glb_u32("oak", &oak),
    );

    // Slightly taller + narrower cone so pine is visibly distinct from oak
    // when both render at once. Same kind of placeholder — the species
    // distinction is what's being demonstrated, not the silhouette.
    let pine = ground_origin(PrimitiveMesh::cone(0.45, 2.6, 16, true), 1.3);
    write_bytes(
        &assets_path(&["models", "trees", "pine.glb"]),
        &encode_glb_u32("pine", &pine),
    );

    // Building proxy: wider/deeper than the original 1 m cube to read as
    // a structure rather than a single stockpiled crate.
    let camp = ground_origin(PrimitiveMesh::cube(Vec3::new(1.6, 1.6, 1.2)), 0.6);
    write_bytes(
        &assets_path(&["models", "buildings", "lumber_camp.glb"]),
        &encode_glb_u32("lumber_camp", &camp),
    );

    // Lumberjack proxy: same 1.6 m capsule the procedural pawn used,
    // origin lifted to its feet.
    let lumberjack = ground_origin(PrimitiveMesh::capsule(0.30, 1.6, 16, 3), 0.8);
    write_bytes(
        &assets_path(&["models", "pawns", "lumberjack.glb"]),
        &encode_glb_u32("lumberjack", &lumberjack),
    );

    // --- assets/textures/** (lumber-camp placeholder PNGs) ----------------
    //
    // 16×16 RGBA8, sRGB-flavoured colour with a small luminance jitter so
    // the texture sample reads as a *texture* rather than a flat constant
    // — drives home that the per-kind albedo is coming from PNG bytes,
    // not from the material's `albedo_factor`.

    write_png(
        &assets_path(&["textures", "trees", "oak_bark.png"]),
        &noisy_solid([105, 70, 40, 255], 16, 12),
    );
    write_png(
        &assets_path(&["textures", "trees", "pine_bark.png"]),
        &noisy_solid([80, 55, 35, 255], 16, 10),
    );
    write_png(
        &assets_path(&["textures", "buildings", "wood_planks.png"]),
        &noisy_solid([150, 100, 60, 255], 16, 18),
    );
    write_png(
        &assets_path(&["textures", "pawns", "lumberjack.png"]),
        &noisy_solid([215, 165, 125, 255], 16, 10),
    );
}

/// Translate every vertex of `mesh` upward by `dz` so what was the
/// `z = -half_height` extent ends up at `z = 0` — i.e. "origin at the base".
/// Sim-side placement code can then plant objects at `Z = floor_height`
/// without any per-kind height offset.
fn ground_origin(mut mesh: PrimitiveMesh, dz: f32) -> PrimitiveMesh {
    for v in &mut mesh.vertices {
        v.position[2] += dz;
    }
    mesh
}

/// Resolve `parts` as a path under the repo's `assets/` directory and
/// ensure its parent exists.
fn assets_path(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets");
    for p in parts {
        path.push(p);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create asset output directory");
    }
    path
}

fn write_bytes(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn write_png(path: &Path, image: &ImageBuffer<Rgba<u8>, Vec<u8>>) {
    image
        .save(path)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({}×{})",
        path.display(),
        image.width(),
        image.height()
    );
}

/// Build an RGBA8 image of `size`×`size` filled with `base`, modulated by
/// a small per-pixel luminance jitter (`±amplitude`). Deterministic hash so
/// re-running the generator produces byte-identical output.
fn noisy_solid(base: [u8; 4], size: u32, amplitude: i32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    ImageBuffer::from_fn(size, size, |x, y| {
        let h = jitter(x, y, size);
        let jitter = (h % (2 * amplitude as i64 + 1) - amplitude as i64) as i32;
        let nudge = |c: u8| ((c as i32 + jitter).clamp(0, 255)) as u8;
        Rgba([nudge(base[0]), nudge(base[1]), nudge(base[2]), base[3]])
    })
}

/// Tiny deterministic hash → integer jitter per (x, y). Inline-rolled rather
/// than pulling in `rand` for one use; the bytes that result are stable as
/// long as this function is.
fn jitter(x: u32, y: u32, size: u32) -> i64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    h ^= (x as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
    h ^= (y as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= (size as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    h ^= h.rotate_left(27);
    (h as i64).wrapping_abs()
}

/// Encode a single-primitive `.glb` with the cube's locked-in `u16`-index
/// layout. Kept as the dedicated path for `assets/test/cube.glb` because
/// the mesh decoder's tests `include_bytes!` it — any byte-level change
/// would surface as test churn even when topology is unchanged.
fn encode_unit_cube_glb(mesh: &PrimitiveMesh) -> Vec<u8> {
    let half = Vec3::ONE * 0.5;
    let pos_min = format!("[{:.6},{:.6},{:.6}]", -half.x, -half.y, -half.z);
    let pos_max = format!("[{:.6},{:.6},{:.6}]", half.x, half.y, half.z);
    let indices_bytes: Vec<u8> = mesh
        .indices
        .iter()
        .flat_map(|&i| {
            let v: u16 = i.try_into().expect("cube indices fit in u16");
            v.to_le_bytes()
        })
        .collect();
    encode_glb_inner(
        "cube",
        mesh,
        COMPONENT_TYPE_U16,
        &indices_bytes,
        &pos_min,
        &pos_max,
    )
}

/// Encode an arbitrary single-primitive mesh as `.glb`. Always uses `u32`
/// indices — fine for every placeholder we emit (decoder reads them via
/// `into_u32()` anyway), and removes the "do indices fit in u16?" branch
/// from the call sites.
fn encode_glb_u32(label: &str, mesh: &PrimitiveMesh) -> Vec<u8> {
    let (pmin, pmax) = position_bounds(mesh);
    let pos_min = format!("[{:.6},{:.6},{:.6}]", pmin.x, pmin.y, pmin.z);
    let pos_max = format!("[{:.6},{:.6},{:.6}]", pmax.x, pmax.y, pmax.z);
    let indices_bytes: Vec<u8> = mesh.indices.iter().flat_map(|&i| i.to_le_bytes()).collect();
    encode_glb_inner(
        label,
        mesh,
        COMPONENT_TYPE_U32,
        &indices_bytes,
        &pos_min,
        &pos_max,
    )
}

fn position_bounds(mesh: &PrimitiveMesh) -> (Vec3, Vec3) {
    let mut min = Vec3::from(z_up_to_y_up(mesh.vertices[0].position));
    let mut max = min;
    for v in &mesh.vertices[1..] {
        let p = Vec3::from(z_up_to_y_up(v.position));
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

/// Inverse of the import-side Y-up→Z-up flip in `decode_gltf_mesh`. The
/// engine's procedural meshes are authored Z-up; glTF's spec-default is
/// Y-up; the loader does `(x, y, z) → (x, -z, y)` on read, so we apply
/// `(x, y, z) → (x, z, -y)` on write to round-trip back to the engine's
/// frame after import. Applied to both positions and normals before they
/// hit the byte stream — UVs aren't axis-bound.
fn z_up_to_y_up(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

/// Shared `.glb` packer: positions, normals, UVs, then indices, four
/// contiguous buffer views into one buffer. The `.glb` JSON + BIN chunks
/// are both padded to 4-byte boundaries per spec.
///
/// `index_component_type` + `index_bytes` are passed in because the cube
/// path uses `u16` and the general path uses `u32`; everything else is
/// identical between them.
fn encode_glb_inner(
    label: &str,
    mesh: &PrimitiveMesh,
    index_component_type: u32,
    index_bytes: &[u8],
    pos_min: &str,
    pos_max: &str,
) -> Vec<u8> {
    let positions_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| {
            z_up_to_y_up(v.position)
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect();
    let normals_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| {
            z_up_to_y_up(v.normal)
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect();
    let uvs_bytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.uv.iter().flat_map(|f| f.to_le_bytes()))
        .collect();

    let positions_len = positions_bytes.len();
    let normals_len = normals_bytes.len();
    let uvs_len = uvs_bytes.len();
    let indices_len = index_bytes.len();

    let positions_offset = 0;
    let normals_offset = positions_offset + positions_len;
    let uvs_offset = normals_offset + normals_len;
    let indices_offset = uvs_offset + uvs_len;
    let buffer_len = indices_offset + indices_len;

    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"currawong gen_test_assets"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"{label}"}}],"meshes":[{{"name":"{label}","primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"mode":{mode}}}]}}],"accessors":[{{"bufferView":0,"componentType":{ct_float},"count":{vcount},"type":"VEC3","min":{pos_min},"max":{pos_max}}},{{"bufferView":1,"componentType":{ct_float},"count":{vcount},"type":"VEC3"}},{{"bufferView":2,"componentType":{ct_float},"count":{vcount},"type":"VEC2"}},{{"bufferView":3,"componentType":{ct_idx},"count":{icount},"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":{p_off},"byteLength":{p_len}}},{{"buffer":0,"byteOffset":{n_off},"byteLength":{n_len}}},{{"buffer":0,"byteOffset":{u_off},"byteLength":{u_len}}},{{"buffer":0,"byteOffset":{i_off},"byteLength":{i_len}}}],"buffers":[{{"byteLength":{buf_len}}}]}}"#,
        label = label,
        mode = MODE_TRIANGLES,
        ct_float = COMPONENT_TYPE_FLOAT,
        ct_idx = index_component_type,
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

    let mut json_bytes = json.into_bytes();
    let json_pad = (4 - json_bytes.len() % 4) % 4;
    json_bytes.extend(std::iter::repeat_n(b' ', json_pad));

    let mut bin_bytes = Vec::with_capacity(buffer_len);
    bin_bytes.extend_from_slice(&positions_bytes);
    bin_bytes.extend_from_slice(&normals_bytes);
    bin_bytes.extend_from_slice(&uvs_bytes);
    bin_bytes.extend_from_slice(index_bytes);
    let bin_pad = (4 - bin_bytes.len() % 4) % 4;
    bin_bytes.extend(std::iter::repeat_n(0u8, bin_pad));

    let total_len: u32 = (12 + 8 + json_bytes.len() + 8 + bin_bytes.len()) as u32;

    let mut out = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total_len.to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
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
