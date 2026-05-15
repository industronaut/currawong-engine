//! CPU-side mesh primitive generators.
//!
//! [`PrimitiveMesh`] is a vertex + index buffer pair in the canonical
//! [`PosNormalUv`] layout, ready to upload to wgpu and draw with any
//! material that demands that layout (today: [`PbrMaterial`](super::PbrMaterial)).
//!
//! ## Conventions
//!
//! - **Z-up, right-handed.** Plane normal is `+Z`; cylinder and cone axes
//!   are along `Z`; every primitive is centred at the origin.
//! - **Winding is CCW from outside**, matching the wgpu default
//!   [`FrontFace::Ccw`](wgpu::FrontFace).
//! - **Per-face normals** on the cube, cylinder caps, and cone base —
//!   flat-shaded. **Smooth normals** on the sphere, cylinder side, and
//!   cone slant — radial / analytic.
//! - **`u32` indices.** Subdividing a sphere to 256×128 overflows `u16`;
//!   uploading as `IndexFormat::Uint32` is the boring choice that never
//!   surprises.
//!
//! ## Scope
//!
//! Cube, plane, sphere, cylinder, cone, capsule. Torus and rounded boxes
//! land when something needs them. Tangent-bearing variants are a separate
//! constructor returning a different vertex layout when a normal-mapped
//! material exists to consume them.

use std::f32::consts::{PI, TAU};

use glam::{UVec2, Vec2, Vec3};

use super::vertex::PosNormalUv;

/// CPU-side mesh data: a `PosNormalUv` vertex buffer plus a `u32` index
/// buffer. Upload both to wgpu and draw with
/// [`IndexFormat::Uint32`](wgpu::IndexFormat::Uint32).
#[derive(Clone, Debug, Default)]
pub struct PrimitiveMesh {
    pub vertices: Vec<PosNormalUv>,
    pub indices: Vec<u32>,
}

impl PrimitiveMesh {
    /// Index count — convenience for the `draw_indexed` range.
    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Axis-aligned box centred at the origin with the given full extents
    /// along `X`, `Y`, and `Z`. Each face is flat-shaded (24 vertices,
    /// 36 indices) with UVs spanning 0..1 across the face.
    ///
    /// `size` components must be positive; zero or negative components
    /// produce a degenerate mesh.
    pub fn cube(size: Vec3) -> Self {
        let h = size * 0.5;

        // (face normal, four corner positions in CCW order seen from outside)
        let faces: [(Vec3, [Vec3; 4]); 6] = [
            // +X
            (
                Vec3::X,
                [
                    Vec3::new(h.x, -h.y, -h.z),
                    Vec3::new(h.x, h.y, -h.z),
                    Vec3::new(h.x, h.y, h.z),
                    Vec3::new(h.x, -h.y, h.z),
                ],
            ),
            // -X
            (
                -Vec3::X,
                [
                    Vec3::new(-h.x, h.y, -h.z),
                    Vec3::new(-h.x, -h.y, -h.z),
                    Vec3::new(-h.x, -h.y, h.z),
                    Vec3::new(-h.x, h.y, h.z),
                ],
            ),
            // +Y
            (
                Vec3::Y,
                [
                    Vec3::new(h.x, h.y, -h.z),
                    Vec3::new(-h.x, h.y, -h.z),
                    Vec3::new(-h.x, h.y, h.z),
                    Vec3::new(h.x, h.y, h.z),
                ],
            ),
            // -Y
            (
                -Vec3::Y,
                [
                    Vec3::new(-h.x, -h.y, -h.z),
                    Vec3::new(h.x, -h.y, -h.z),
                    Vec3::new(h.x, -h.y, h.z),
                    Vec3::new(-h.x, -h.y, h.z),
                ],
            ),
            // +Z (top)
            (
                Vec3::Z,
                [
                    Vec3::new(-h.x, -h.y, h.z),
                    Vec3::new(h.x, -h.y, h.z),
                    Vec3::new(h.x, h.y, h.z),
                    Vec3::new(-h.x, h.y, h.z),
                ],
            ),
            // -Z (bottom)
            (
                -Vec3::Z,
                [
                    Vec3::new(-h.x, h.y, -h.z),
                    Vec3::new(h.x, h.y, -h.z),
                    Vec3::new(h.x, -h.y, -h.z),
                    Vec3::new(-h.x, -h.y, -h.z),
                ],
            ),
        ];
        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for (face_idx, (normal, corners)) in faces.iter().enumerate() {
            let base = (face_idx * 4) as u32;
            for (corner, uv) in corners.iter().zip(uvs.iter()) {
                vertices.push(PosNormalUv {
                    position: corner.to_array(),
                    normal: normal.to_array(),
                    uv: *uv,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self { vertices, indices }
    }

    /// Subdivided quad on the XY plane, centred at the origin, with the
    /// normal pointing `+Z`. `size` is the full extent along X and Y;
    /// `segments` is the quad count along each axis (clamped to `>= 1`).
    /// UVs span 0..1 across the whole plane.
    pub fn plane(size: Vec2, segments: UVec2) -> Self {
        let sx = segments.x.max(1);
        let sy = segments.y.max(1);
        let inv_sx = 1.0 / sx as f32;
        let inv_sy = 1.0 / sy as f32;
        let half = size * 0.5;

        let nx = sx + 1;
        let ny = sy + 1;
        let mut vertices = Vec::with_capacity((nx * ny) as usize);
        for j in 0..ny {
            for i in 0..nx {
                let u = i as f32 * inv_sx;
                let v = j as f32 * inv_sy;
                vertices.push(PosNormalUv {
                    position: [u * size.x - half.x, v * size.y - half.y, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [u, v],
                });
            }
        }

        let mut indices = Vec::with_capacity((sx * sy * 6) as usize);
        for j in 0..sy {
            for i in 0..sx {
                let i0 = j * nx + i;
                let i1 = i0 + 1;
                let i2 = i0 + nx;
                let i3 = i2 + 1;
                // CCW from +Z
                indices.extend_from_slice(&[i0, i1, i3, i0, i3, i2]);
            }
        }
        Self { vertices, indices }
    }

    /// UV sphere centred at the origin. `longitudes` is the azimuthal
    /// segment count around the Z axis; `latitudes` is the polar segment
    /// count from `+Z` pole to `-Z` pole. Both are clamped to at least 3
    /// and 2 respectively. Normals are radial (smooth); UVs are an
    /// equirectangular projection — `u = θ / 2π`, `v = 1 - φ / π`.
    ///
    /// The seam at `θ = 0` is duplicated so the texture wraps cleanly;
    /// poles collapse to a ring of vertices sharing position but with
    /// distinct UVs.
    pub fn sphere(radius: f32, longitudes: u32, latitudes: u32) -> Self {
        let lon = longitudes.max(3);
        let lat = latitudes.max(2);

        let n_ring = lon + 1; // seam duplicated
        let n_rows = lat + 1; // pole to pole
        let mut vertices = Vec::with_capacity((n_ring * n_rows) as usize);
        for j in 0..n_rows {
            let v = j as f32 / lat as f32;
            let phi = v * PI;
            let (sphi, cphi) = phi.sin_cos();
            for i in 0..n_ring {
                let u = i as f32 / lon as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                let normal = Vec3::new(sphi * ctheta, sphi * stheta, cphi);
                vertices.push(PosNormalUv {
                    position: (normal * radius).to_array(),
                    normal: normal.to_array(),
                    uv: [u, 1.0 - v],
                });
            }
        }

        let mut indices = Vec::with_capacity((lon * lat * 6) as usize);
        for j in 0..lat {
            for i in 0..lon {
                let i0 = j * n_ring + i;
                let i1 = i0 + 1;
                let i2 = i0 + n_ring;
                let i3 = i2 + 1;
                // CCW from outside (positive φ is downward in our convention,
                // so j+1 is the "next row down").
                indices.extend_from_slice(&[i0, i3, i1, i0, i2, i3]);
            }
        }
        Self { vertices, indices }
    }

    /// Cylinder with its axis along `Z`, centred at the origin so the
    /// caps sit at `z = ±height/2`. `segments` is the azimuthal segment
    /// count (clamped to >= 3). With `caps = true`, top and bottom disks
    /// are added with flat ±Z normals.
    ///
    /// Side normals are radial (smooth); the side UV maps `θ` to `u` and
    /// height to `v` (bottom = 0, top = 1). The seam at `θ = 0` is
    /// duplicated.
    pub fn cylinder(radius: f32, height: f32, segments: u32, caps: bool) -> Self {
        let seg = segments.max(3);
        let half_h = height * 0.5;
        let n_ring = seg + 1;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // --- Side: two rings, seam-duplicated ---
        for ring in 0..2 {
            let z = if ring == 0 { -half_h } else { half_h };
            let v = ring as f32; // 0 at bottom, 1 at top
            for i in 0..n_ring {
                let u = i as f32 / seg as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                vertices.push(PosNormalUv {
                    position: [radius * ctheta, radius * stheta, z],
                    normal: [ctheta, stheta, 0.0],
                    uv: [u, v],
                });
            }
        }
        // Side quads, CCW from outside.
        for i in 0..seg {
            let b0 = i;
            let b1 = i + 1;
            let t0 = i + n_ring;
            let t1 = t0 + 1;
            indices.extend_from_slice(&[b0, b1, t1, b0, t1, t0]);
        }

        if caps {
            // Top cap. Centre vertex + ring with normal +Z.
            let top_centre = vertices.len() as u32;
            vertices.push(PosNormalUv {
                position: [0.0, 0.0, half_h],
                normal: [0.0, 0.0, 1.0],
                uv: [0.5, 0.5],
            });
            let top_ring_base = vertices.len() as u32;
            for i in 0..n_ring {
                let u = i as f32 / seg as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                vertices.push(PosNormalUv {
                    position: [radius * ctheta, radius * stheta, half_h],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.5 + 0.5 * ctheta, 0.5 - 0.5 * stheta],
                });
            }
            for i in 0..seg {
                let a = top_ring_base + i;
                let b = top_ring_base + i + 1;
                // CCW seen from +Z.
                indices.extend_from_slice(&[top_centre, a, b]);
            }

            // Bottom cap. Centre vertex + ring with normal -Z.
            let bot_centre = vertices.len() as u32;
            vertices.push(PosNormalUv {
                position: [0.0, 0.0, -half_h],
                normal: [0.0, 0.0, -1.0],
                uv: [0.5, 0.5],
            });
            let bot_ring_base = vertices.len() as u32;
            for i in 0..n_ring {
                let u = i as f32 / seg as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                vertices.push(PosNormalUv {
                    position: [radius * ctheta, radius * stheta, -half_h],
                    normal: [0.0, 0.0, -1.0],
                    uv: [0.5 + 0.5 * ctheta, 0.5 + 0.5 * stheta],
                });
            }
            for i in 0..seg {
                let a = bot_ring_base + i;
                let b = bot_ring_base + i + 1;
                // CCW seen from -Z (i.e. reversed compared to the top).
                indices.extend_from_slice(&[bot_centre, b, a]);
            }
        }

        Self { vertices, indices }
    }

    /// Cone with its axis along `Z`, centred at the origin so the apex is
    /// at `z = +height/2` and the base at `z = -height/2`. `segments` is
    /// the azimuthal segment count (clamped to >= 3). With `base = true`,
    /// the bottom disk is added with a flat `-Z` normal.
    ///
    /// Slant normals are the analytic cone normal `normalize((h·cosθ,
    /// h·sinθ, r))`, constant along the slant for a given azimuth. Each
    /// slant face gets its own apex vertex whose normal is the midpoint
    /// of the two adjacent base normals — smooth shading without the
    /// usual apex singularity artefacts.
    pub fn cone(radius: f32, height: f32, segments: u32, base: bool) -> Self {
        let seg = segments.max(3);
        let half_h = height * 0.5;

        // The slant normal at azimuth θ is normalize((h·cosθ, h·sinθ, r)).
        let slant_normal = |theta: f32| -> Vec3 {
            let (s, c) = theta.sin_cos();
            Vec3::new(height * c, height * s, radius).normalize()
        };

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // --- Slant ---
        // Base ring: seg + 1 vertices (seam duplicated). Apex: one
        // vertex per slant face (seg of them), each with the normal at
        // the midpoint of the corresponding base-edge angles.
        let n_ring = seg + 1;
        let base_ring_start = 0;
        for i in 0..n_ring {
            let u = i as f32 / seg as f32;
            let theta = u * TAU;
            let (stheta, ctheta) = theta.sin_cos();
            vertices.push(PosNormalUv {
                position: [radius * ctheta, radius * stheta, -half_h],
                normal: slant_normal(theta).to_array(),
                uv: [u, 0.0],
            });
        }
        let apex_start = vertices.len() as u32;
        for i in 0..seg {
            let theta_mid = (i as f32 + 0.5) / seg as f32 * TAU;
            let u = (i as f32 + 0.5) / seg as f32;
            vertices.push(PosNormalUv {
                position: [0.0, 0.0, half_h],
                normal: slant_normal(theta_mid).to_array(),
                uv: [u, 1.0],
            });
        }
        for i in 0..seg {
            let b0 = base_ring_start + i;
            let b1 = b0 + 1;
            let apex = apex_start + i;
            // CCW from outside.
            indices.extend_from_slice(&[b0, b1, apex]);
        }

        if base {
            // Bottom disk: centre + seam-duplicated ring with normal -Z.
            let centre = vertices.len() as u32;
            vertices.push(PosNormalUv {
                position: [0.0, 0.0, -half_h],
                normal: [0.0, 0.0, -1.0],
                uv: [0.5, 0.5],
            });
            let ring_start = vertices.len() as u32;
            for i in 0..n_ring {
                let u = i as f32 / seg as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                vertices.push(PosNormalUv {
                    position: [radius * ctheta, radius * stheta, -half_h],
                    normal: [0.0, 0.0, -1.0],
                    uv: [0.5 + 0.5 * ctheta, 0.5 + 0.5 * stheta],
                });
            }
            for i in 0..seg {
                let a = ring_start + i;
                let b = ring_start + i + 1;
                // CCW seen from -Z.
                indices.extend_from_slice(&[centre, b, a]);
            }
        }

        Self { vertices, indices }
    }

    /// Capsule (rounded cylinder) with its axis along `Z`, centred at the
    /// origin so the hemispherical caps touch `z = ±height/2`.
    ///
    /// - `radius` is the radius of the cylindrical section and the two
    ///   hemispherical caps.
    /// - `height` is the **total** length along `Z`. The straight
    ///   cylindrical section has height `(height - 2·radius).max(0)`; if
    ///   `height < 2·radius` the section collapses and the shape
    ///   degenerates to a sphere (no panic — useful default for
    ///   prototyping).
    /// - `longitudes` is the azimuthal segment count (clamped to >= 3),
    ///   matching [`Self::sphere`].
    /// - `cap_latitudes` is the polar segment count for **one** hemisphere
    ///   (clamped to >= 1). A full-sphere equivalent has `2 ·
    ///   cap_latitudes` total latitudes.
    ///
    /// Normals are smooth everywhere: hemisphere normals are radial, the
    /// cylinder side is radial, and at the equator the two meet exactly
    /// (`sin(π/2) = 1`, `cos(π/2) = 0`) — no shading seam. UVs map `θ` to
    /// `u` and surface-arc-length-from-the-top-pole to `v`, so a checker
    /// texture doesn't compress at the caps.
    pub fn capsule(radius: f32, height: f32, longitudes: u32, cap_latitudes: u32) -> Self {
        let lon = longitudes.max(3);
        let lat = cap_latitudes.max(1);
        let cyl_h = (height - 2.0 * radius).max(0.0);
        let half_cyl_h = cyl_h * 0.5;
        let total_arc = radius * PI + cyl_h;

        let n_ring = lon + 1; // seam duplicated
        let n_rows = 2 * lat + 2; // top pole..top equator + bottom equator..bottom pole

        // Per-row data: (z, ring radius, normal-z = cos(phi), v).
        let mut row_data: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(n_rows as usize);
        // Top hemisphere: φ from 0 (top pole) to π/2 (top equator).
        for k in 0..=lat {
            let t = k as f32 / lat as f32;
            let phi = t * (PI * 0.5);
            let (sphi, cphi) = phi.sin_cos();
            let arc = radius * phi;
            let v = if total_arc > 0.0 {
                arc / total_arc
            } else {
                0.0
            };
            row_data.push((half_cyl_h + radius * cphi, radius * sphi, cphi, v));
        }
        // Bottom hemisphere: φ from π/2 (bottom equator) to π (bottom pole).
        for k in 0..=lat {
            let t = k as f32 / lat as f32;
            let phi = PI * 0.5 + t * (PI * 0.5);
            let (sphi, cphi) = phi.sin_cos();
            let arc = radius * (PI * 0.5) + cyl_h + radius * (phi - PI * 0.5);
            let v = if total_arc > 0.0 {
                arc / total_arc
            } else {
                1.0
            };
            row_data.push((-half_cyl_h + radius * cphi, radius * sphi, cphi, v));
        }

        let mut vertices = Vec::with_capacity((n_rows * n_ring) as usize);
        for &(z, ring_radius, nz, v) in &row_data {
            // sin(phi) is non-negative for phi in [0, π], so recover it from cos.
            let sphi = (1.0 - nz * nz).max(0.0).sqrt();
            for j in 0..n_ring {
                let u = j as f32 / lon as f32;
                let theta = u * TAU;
                let (stheta, ctheta) = theta.sin_cos();
                vertices.push(PosNormalUv {
                    position: [ring_radius * ctheta, ring_radius * stheta, z],
                    normal: [sphi * ctheta, sphi * stheta, nz],
                    uv: [u, v],
                });
            }
        }

        let mut indices = Vec::with_capacity(((n_rows - 1) * lon * 6) as usize);
        for i in 0..n_rows - 1 {
            for j in 0..lon {
                let i0 = i * n_ring + j;
                let i1 = i0 + 1;
                let i2 = i0 + n_ring;
                let i3 = i2 + 1;
                // Same winding as Self::sphere — CCW from outside.
                indices.extend_from_slice(&[i0, i3, i1, i0, i2, i3]);
            }
        }

        Self { vertices, indices }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices_in_bounds(mesh: &PrimitiveMesh) -> bool {
        let n = mesh.vertices.len() as u32;
        mesh.indices.iter().all(|&i| i < n)
    }

    fn vertex_count_is_multiple_of_three(mesh: &PrimitiveMesh) -> bool {
        mesh.indices.len().is_multiple_of(3)
    }

    // --- cube ---

    #[test]
    fn cube_has_24_verts_and_36_indices() {
        let m = PrimitiveMesh::cube(Vec3::ONE);
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
        assert!(indices_in_bounds(&m));
        assert!(vertex_count_is_multiple_of_three(&m));
    }

    #[test]
    fn cube_extents_match_requested_size() {
        let size = Vec3::new(2.0, 4.0, 6.0);
        let m = PrimitiveMesh::cube(size);
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in &m.vertices {
            let p = Vec3::from_array(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        assert_eq!(min, -size * 0.5);
        assert_eq!(max, size * 0.5);
    }

    #[test]
    fn cube_face_normals_point_outward() {
        // Each face's four verts share a normal that is one of ±X, ±Y, ±Z.
        let m = PrimitiveMesh::cube(Vec3::ONE);
        for face in 0..6 {
            let n = Vec3::from_array(m.vertices[face * 4].normal);
            assert!(
                (n - Vec3::X).length() < 1e-5
                    || (n + Vec3::X).length() < 1e-5
                    || (n - Vec3::Y).length() < 1e-5
                    || (n + Vec3::Y).length() < 1e-5
                    || (n - Vec3::Z).length() < 1e-5
                    || (n + Vec3::Z).length() < 1e-5,
                "face {face} normal {n:?} is not axis-aligned"
            );
            for k in 1..4 {
                assert_eq!(m.vertices[face * 4 + k].normal, n.to_array());
            }
        }
    }

    // --- plane ---

    #[test]
    fn plane_default_has_one_quad() {
        let m = PrimitiveMesh::plane(Vec2::ONE, UVec2::ONE);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 6);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn plane_subdivided_counts_match_grid() {
        let m = PrimitiveMesh::plane(Vec2::new(2.0, 4.0), UVec2::new(3, 5));
        assert_eq!(m.vertices.len(), 4 * 6);
        assert_eq!(m.indices.len(), 3 * 5 * 6);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn plane_all_normals_are_plus_z() {
        let m = PrimitiveMesh::plane(Vec2::ONE, UVec2::new(2, 2));
        for v in &m.vertices {
            assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn plane_zero_segments_clamps_to_one() {
        let m = PrimitiveMesh::plane(Vec2::ONE, UVec2::ZERO);
        assert_eq!(m.vertices.len(), 4);
        assert_eq!(m.indices.len(), 6);
    }

    // --- sphere ---

    #[test]
    fn sphere_vertex_positions_have_radius_length() {
        let r = 2.5;
        let m = PrimitiveMesh::sphere(r, 16, 8);
        for v in &m.vertices {
            let p = Vec3::from_array(v.position);
            assert!((p.length() - r).abs() < 1e-4, "vertex off-radius: {p:?}");
        }
    }

    #[test]
    fn sphere_normals_match_normalized_position() {
        let m = PrimitiveMesh::sphere(3.0, 12, 6);
        for v in &m.vertices {
            let p = Vec3::from_array(v.position);
            let n = Vec3::from_array(v.normal);
            assert!((p.normalize() - n).length() < 1e-4);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn sphere_indices_in_bounds_and_complete() {
        let m = PrimitiveMesh::sphere(1.0, 16, 8);
        assert!(indices_in_bounds(&m));
        assert!(vertex_count_is_multiple_of_three(&m));
        assert_eq!(m.indices.len(), 16 * 8 * 6);
    }

    // --- cylinder ---

    #[test]
    fn cylinder_without_caps_has_only_sides() {
        let m = PrimitiveMesh::cylinder(1.0, 2.0, 8, false);
        // 2 rings × 9 verts (seam dup) = 18.
        assert_eq!(m.vertices.len(), 18);
        // 8 quads × 6 = 48.
        assert_eq!(m.indices.len(), 48);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn cylinder_with_caps_adds_two_fans() {
        let m = PrimitiveMesh::cylinder(1.0, 2.0, 8, true);
        // 18 side + 2 × (1 centre + 9 ring) = 18 + 20 = 38.
        assert_eq!(m.vertices.len(), 38);
        // 48 sides + 2 × (8 fan tris × 3 indices) = 48 + 48 = 96.
        assert_eq!(m.indices.len(), 96);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn cylinder_side_normals_are_radial_and_horizontal() {
        let m = PrimitiveMesh::cylinder(1.0, 2.0, 8, false);
        for v in &m.vertices {
            let n = Vec3::from_array(v.normal);
            assert!((n.z).abs() < 1e-5, "side normal has Z component: {n:?}");
            assert!((n.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn cylinder_height_matches_request() {
        let m = PrimitiveMesh::cylinder(1.0, 3.0, 6, true);
        let zs: Vec<f32> = m.vertices.iter().map(|v| v.position[2]).collect();
        let min = zs.iter().copied().fold(f32::INFINITY, f32::min);
        let max = zs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((min + 1.5).abs() < 1e-5);
        assert!((max - 1.5).abs() < 1e-5);
    }

    // --- cone ---

    #[test]
    fn cone_slant_only_counts() {
        let m = PrimitiveMesh::cone(1.0, 2.0, 8, false);
        // (segments + 1) base ring + segments apex verts = 9 + 8 = 17.
        assert_eq!(m.vertices.len(), 17);
        // 8 slant tris × 3 indices = 24.
        assert_eq!(m.indices.len(), 24);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn cone_with_base_adds_disk() {
        let m = PrimitiveMesh::cone(1.0, 2.0, 8, true);
        // 17 slant + 1 centre + 9 ring = 27.
        assert_eq!(m.vertices.len(), 27);
        // 24 slant + 8 fan × 3 = 48.
        assert_eq!(m.indices.len(), 48);
        assert!(indices_in_bounds(&m));
    }

    #[test]
    fn cone_apex_is_at_top() {
        let m = PrimitiveMesh::cone(1.0, 2.0, 6, false);
        // First (segments + 1) verts are the base ring; the next `segments`
        // are apex copies. All apex positions equal (0, 0, +1).
        let apex_start = 7;
        for i in 0..6 {
            assert_eq!(m.vertices[apex_start + i].position, [0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn cone_slant_normals_are_unit_length() {
        let m = PrimitiveMesh::cone(1.0, 2.0, 12, true);
        for v in &m.vertices {
            let n = Vec3::from_array(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }
    }

    // --- capsule ---

    #[test]
    fn capsule_vertex_and_index_counts() {
        // lon = 8, cap_lat = 2 → rows = 2*2 + 2 = 6; ring verts = 9.
        // Verts = 6 × 9 = 54. Quads = (6 - 1) × 8 = 40; indices = 240.
        let m = PrimitiveMesh::capsule(0.5, 2.0, 8, 2);
        assert_eq!(m.vertices.len(), 54);
        assert_eq!(m.indices.len(), 240);
        assert!(indices_in_bounds(&m));
        assert!(vertex_count_is_multiple_of_three(&m));
    }

    #[test]
    fn capsule_total_height_matches_request() {
        let r = 0.4;
        let h = 2.0;
        let m = PrimitiveMesh::capsule(r, h, 16, 4);
        let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &m.vertices {
            zmin = zmin.min(v.position[2]);
            zmax = zmax.max(v.position[2]);
        }
        assert!((zmin + h * 0.5).abs() < 1e-5, "zmin={zmin}");
        assert!((zmax - h * 0.5).abs() < 1e-5, "zmax={zmax}");
    }

    #[test]
    fn capsule_cylinder_radius_bound() {
        // Every vertex must be inside or on the capsule's swept volume:
        // radial distance from axis is at most `radius`.
        let r = 0.6;
        let m = PrimitiveMesh::capsule(r, 2.0, 12, 3);
        for v in &m.vertices {
            let radial = (v.position[0] * v.position[0] + v.position[1] * v.position[1]).sqrt();
            assert!(radial <= r + 1e-5, "radial={radial} exceeds radius={r}");
        }
    }

    #[test]
    fn capsule_normals_unit_length() {
        let m = PrimitiveMesh::capsule(0.5, 2.0, 12, 3);
        for v in &m.vertices {
            let n = Vec3::from_array(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn capsule_equator_normals_are_purely_radial() {
        // The seam between cap and cylinder sits at the top equator (row
        // `cap_lat`) and the bottom equator (row `cap_lat + 1`). Both
        // rings must have z-component = 0 so the hemisphere blends
        // smoothly into the cylinder side.
        let lon: u32 = 8;
        let cap_lat: u32 = 2;
        let n_ring = lon + 1;
        let m = PrimitiveMesh::capsule(0.5, 2.0, lon, cap_lat);
        for &row in &[cap_lat, cap_lat + 1] {
            for j in 0..n_ring {
                let idx = (row * n_ring + j) as usize;
                let nz = m.vertices[idx].normal[2];
                assert!(nz.abs() < 1e-5, "row {row} vertex {j} has nz={nz}");
            }
        }
    }

    #[test]
    fn capsule_height_below_2r_clamps_to_sphere_shape() {
        // height < 2r: cylinder section collapses, the shape is a sphere
        // of the given radius. Total Z extent should be ±radius, not
        // ±height/2 — the requested height is "minimum total length" in
        // this degenerate case.
        let r = 1.0;
        let m = PrimitiveMesh::capsule(r, 0.5, 8, 3);
        let mut zmax = f32::NEG_INFINITY;
        for v in &m.vertices {
            zmax = zmax.max(v.position[2]);
        }
        assert!((zmax - r).abs() < 1e-5, "zmax={zmax}, expected ~{r}");
    }
}
