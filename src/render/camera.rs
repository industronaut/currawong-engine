//! Perspective camera helper for views that render in 3D world space.

use glam::{Mat4, Vec3};

/// Perspective camera helper for views that render in 3D world space.
///
/// Held by the user's [`View`](crate::View) (a View doesn't have to use one —
/// UI/2D views can render directly in clip space). The user updates
/// `position`, `target`, and `aspect` over time; pipelines upload
/// [`Camera::view_proj`] to a uniform buffer each frame.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Right-handed view matrix. Uses `target` as the look-at point.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Right-handed perspective matrix with depth in `[0, 1]` (wgpu/Vulkan
    /// convention, not OpenGL's `[-1, 1]`).
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_radians, self.aspect, self.near, self.far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}
