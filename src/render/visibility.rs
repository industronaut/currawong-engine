//! Visibility primitives: axis-aligned bounding boxes and view-frustum
//! tests. Used by render-object templates ([`visual_bounds`](super::RenderTemplate::visual_bounds))
//! and by the live-instance reconciler
//! ([`RenderInstances::cull`](super::RenderInstances::cull)).

use glam::{Mat4, Vec3, Vec4};

/// Axis-aligned bounding box in some local or world frame.
///
/// Used by render-object templates to declare their *visual* extent —
/// which may include emitter reach and other effects, and so be
/// substantially larger than the template's geometric footprint. See
/// CLAUDE.md's "Visual bounds differ from sim bounds" invariant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// AABB centred at the origin with `half_extent` along each axis.
    pub fn cube(half_extent: f32) -> Self {
        let h = half_extent.abs();
        Self {
            min: Vec3::splat(-h),
            max: Vec3::splat(h),
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half-extents along each axis.
    pub fn extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// AABB enclosing both inputs.
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Transform all 8 corners by `m` and return the AABB of the
    /// transformed points. Conservative: looser than the true bounding
    /// volume of the rotated original (especially under rotation), but
    /// cheap and right for visibility culling.
    pub fn transformed(&self, m: Mat4) -> Self {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let mut min = m.transform_point3(corners[0]);
        let mut max = min;
        for c in &corners[1..] {
            let p = m.transform_point3(*c);
            min = min.min(p);
            max = max.max(p);
        }
        Self { min, max }
    }
}

/// View frustum as 6 planes in `(nx, ny, nz, d)` form. A point `p` is
/// inside the frustum when `nx*p.x + ny*p.y + nz*p.z + d >= 0` for every
/// plane.
///
/// Extracted from a view-projection matrix using the Gribb-Hartmann
/// method, with planes oriented for wgpu / Vulkan / D3D clip space
/// (depth range `[0, 1]`, not OpenGL's `[-1, 1]`).
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Order: left, right, bottom, top, near, far.
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix. Planes are
    /// normalised so distances are in world units.
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let r = view_proj.row(0);
        let s = view_proj.row(1);
        let t = view_proj.row(2);
        let w = view_proj.row(3);

        // Near plane is just `t` (not `w + t`) because wgpu depth is
        // `[0, 1]`: the near-plane inequality is `z >= 0`, not `z >= -w`.
        let planes_unnormalised = [
            w + r, // left:   w + x >= 0
            w - r, // right:  w - x >= 0
            w + s, // bottom: w + y >= 0
            w - s, // top:    w - y >= 0
            t,     // near:   z >= 0
            w - t, // far:    w - z >= 0
        ];
        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in planes_unnormalised.iter().enumerate() {
            let n = Vec3::new(p.x, p.y, p.z).length();
            planes[i] = *p / n.max(f32::MIN_POSITIVE);
        }
        Self { planes }
    }

    /// True if the AABB is at least partially inside the frustum.
    ///
    /// Conservative reject: returns false only when the entire box is
    /// outside at least one plane. Boxes that straddle a plane or are
    /// near-but-outside the frustum corner may return true; that's fine
    /// for culling (false negatives would cause objects to vanish, false
    /// positives just cost a redundant draw).
    pub fn contains_aabb(&self, aabb: &Aabb) -> bool {
        let center = aabb.center();
        let extent = aabb.extents();
        for plane in &self.planes {
            // Signed distance from the AABB centre to the plane.
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            // Half-width of the AABB projected onto the plane normal.
            let radius =
                extent.x * plane.x.abs() + extent.y * plane.y.abs() + extent.z * plane.z.abs();
            if dist + radius < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_center_and_extents() {
        let a = Aabb::new(Vec3::new(-1.0, 0.0, -2.0), Vec3::new(3.0, 4.0, 2.0));
        assert_eq!(a.center(), Vec3::new(1.0, 2.0, 0.0));
        assert_eq!(a.extents(), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn cube_is_symmetric_about_origin() {
        let c = Aabb::cube(0.5);
        assert_eq!(c.min, Vec3::splat(-0.5));
        assert_eq!(c.max, Vec3::splat(0.5));
    }

    #[test]
    fn union_expands_to_enclose_both() {
        let a = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = Aabb::new(Vec3::new(-2.0, 3.0, -1.0), Vec3::new(0.5, 4.0, 0.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::new(-2.0, 0.0, -1.0));
        assert_eq!(u.max, Vec3::new(1.0, 4.0, 1.0));
    }

    #[test]
    fn transformed_translation_only_shifts_min_max() {
        let a = Aabb::cube(0.5);
        let t = a.transformed(Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0)));
        assert!((t.min - Vec3::new(9.5, 19.5, 29.5)).length() < 1e-4);
        assert!((t.max - Vec3::new(10.5, 20.5, 30.5)).length() < 1e-4);
    }

    #[test]
    fn transformed_rotation_grows_aabb() {
        // A unit cube rotated 45° around Y has a larger XZ AABB than the
        // original — the corners stick out further along the rotated diagonals.
        let a = Aabb::cube(0.5);
        let r = a.transformed(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_4));
        // Diagonal across X and Z is sqrt(2)/2 ≈ 0.707 — larger than the
        // original 0.5 half-extent.
        assert!(r.extents().x > 0.7 && r.extents().x < 0.71);
        assert!(r.extents().z > 0.7 && r.extents().z < 0.71);
        // Y is unaffected by rotation around Y.
        assert!((r.extents().y - 0.5).abs() < 1e-4);
    }

    fn perspective(aspect: f32) -> Mat4 {
        Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.1, 100.0)
    }

    fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
        Mat4::look_at_rh(eye, target, Vec3::Y)
    }

    #[test]
    fn frustum_accepts_box_at_origin_with_camera_back() {
        // Camera at (0, 0, 5) looking at origin: a unit cube at origin is
        // squarely in view.
        let vp = perspective(1.0) * look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let f = Frustum::from_view_proj(vp);
        assert!(f.contains_aabb(&Aabb::cube(0.5)));
    }

    #[test]
    fn frustum_rejects_box_behind_camera() {
        // Camera at (0, 0, 5) looking at origin: a cube at z = +20 is
        // behind the camera.
        let vp = perspective(1.0) * look_at(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO);
        let f = Frustum::from_view_proj(vp);
        let behind = Aabb::new(Vec3::new(-0.5, -0.5, 19.5), Vec3::new(0.5, 0.5, 20.5));
        assert!(!f.contains_aabb(&behind));
    }

    #[test]
    fn frustum_rejects_box_outside_side() {
        // Camera at origin looking -Z: a cube at (100, 0, -5) is way
        // outside the side.
        let vp = perspective(1.0) * look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let f = Frustum::from_view_proj(vp);
        let far_right = Aabb::new(Vec3::new(99.5, -0.5, -5.5), Vec3::new(100.5, 0.5, -4.5));
        assert!(!f.contains_aabb(&far_right));
    }

    #[test]
    fn frustum_rejects_box_past_far_plane() {
        // far = 100; a box at z = -200 (i.e., 200m in front of camera) is
        // beyond the far plane.
        let vp = perspective(1.0) * look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let f = Frustum::from_view_proj(vp);
        let too_far = Aabb::new(Vec3::new(-0.5, -0.5, -200.5), Vec3::new(0.5, 0.5, -199.5));
        assert!(!f.contains_aabb(&too_far));
    }
}
