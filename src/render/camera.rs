//! Perspective camera helper for views that render in 3D world space.

use glam::{Mat4, Vec3, Vec4};

use crate::sim::ZoneId;

/// Perspective camera helper for views that render in 3D world space.
///
/// Held by the user's [`View`](crate::View) (a View doesn't have to use one —
/// UI/2D views can render directly in clip space). The user updates
/// `position`, `target`, and `aspect` over time; pipelines upload
/// [`Camera::view_proj`] to a uniform buffer each frame.
///
/// `zone` names the zone the camera is currently looking at. It's
/// `Option<ZoneId>` because UI/2D views don't have a notion of "active
/// zone"; world-space views forward this through
/// [`View::active_zone`](crate::View::active_zone) so the engine knows
/// which zone to ask for environment extraction, visibility culling, and
/// (later) which terrain chunks to draw.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub zone: Option<ZoneId>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, -5.0, 2.0),
            target: Vec3::ZERO,
            up: Vec3::Z,
            fov_y_radians: 60_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
            zone: None,
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

    /// World-space `(right, up)` basis vectors of the camera. Useful for
    /// camera-facing billboards: a particle/sprite vertex shader can expand a
    /// 2D corner offset into a world-space quad as
    /// `pos + right * corner.x * size + up * corner.y * size`.
    ///
    /// The inverse of a pure rotation+translation view matrix has the
    /// camera's world-space axes in its first three columns.
    pub fn billboard_basis(&self) -> (Vec3, Vec3) {
        let view_inv = self.view_matrix().inverse();
        (view_inv.x_axis.truncate(), view_inv.y_axis.truncate())
    }

    /// View-space depths (positive distances along the camera's forward axis)
    /// where each of four CSM cascades ends. Cascade `i` covers the
    /// camera-frustum slice from `splits[i-1]` (or `Camera::near` for `i == 0`)
    /// to `splits[i]`. `splits[3]` always equals `Camera::far`.
    ///
    /// `lambda` blends between *uniform* splits (`0.0`) and *logarithmic*
    /// splits (`1.0`); `0.75` is the standard "practical" value documented in
    /// the GPU Gems 3 CSM article — it gives near cascades enough resolution
    /// without crushing the far ones.
    pub fn cascade_split_distances(&self, lambda: f32) -> [f32; 4] {
        let n = self.near;
        let f = self.far;
        let ratio = f / n;
        let mut splits = [0.0; 4];
        for i in 1..=4usize {
            let frac = i as f32 / 4.0;
            let p_log = n * ratio.powf(frac);
            let p_uniform = n + (f - n) * frac;
            splits[i - 1] = lambda * p_log + (1.0 - lambda) * p_uniform;
        }
        splits
    }

    /// Fit four orthographic light-space view-projection matrices, one per
    /// CSM cascade, each containing the camera-frustum slice it's
    /// responsible for. The matrices map world-space positions into wgpu
    /// clip space (xy ∈ [-1, 1], z ∈ [0, 1]) and are stored straight into
    /// the scene uniform for lit shaders to sample.
    ///
    /// `sun_direction` is a unit vector pointing *from the world toward the
    /// sun* — same convention as
    /// [`ViewEnvironment::sun_direction`](super::environment::ViewEnvironment::sun_direction).
    /// `splits` is the output of [`Camera::cascade_split_distances`].
    ///
    /// Texel-grid snapping (to eliminate edge shimmer under camera motion) is
    /// a future-direction refinement; this routine performs a clean ortho
    /// fit only.
    pub fn fit_shadow_cascades(&self, sun_direction: Vec3, splits: [f32; 4]) -> [Mat4; 4] {
        let view_inv = self.view_matrix().inverse();
        let tan_half_fov = (self.fov_y_radians * 0.5).tan();
        // Sun-direction can be aligned with world up (Z). Pick a non-parallel
        // up vector to avoid look_at_rh degeneracy.
        let up = if sun_direction.z.abs() > 0.9 {
            Vec3::Y
        } else {
            Vec3::Z
        };

        std::array::from_fn(|i| {
            let near_d = if i == 0 { self.near } else { splits[i - 1] };
            let far_d = splits[i];

            let half_h_n = near_d * tan_half_fov;
            let half_w_n = half_h_n * self.aspect;
            let half_h_f = far_d * tan_half_fov;
            let half_w_f = half_h_f * self.aspect;

            // Eight slice corners in view space (camera looks down -Z in RH
            // view space, so z is negative).
            let corners_view = [
                Vec3::new(-half_w_n, -half_h_n, -near_d),
                Vec3::new(half_w_n, -half_h_n, -near_d),
                Vec3::new(-half_w_n, half_h_n, -near_d),
                Vec3::new(half_w_n, half_h_n, -near_d),
                Vec3::new(-half_w_f, -half_h_f, -far_d),
                Vec3::new(half_w_f, -half_h_f, -far_d),
                Vec3::new(-half_w_f, half_h_f, -far_d),
                Vec3::new(half_w_f, half_h_f, -far_d),
            ];

            let corners_world: [Vec3; 8] =
                std::array::from_fn(|k| view_inv.transform_point3(corners_view[k]));

            let focus = corners_world.iter().copied().sum::<Vec3>() / 8.0;
            let light_view = Mat4::look_at_rh(focus + sun_direction, focus, up);

            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for p_w in &corners_world {
                let p_l = light_view.transform_point3(*p_w);
                min = min.min(p_l);
                max = max.max(p_l);
            }

            // glam orthographic_rh maps view-space z in [-near, -far] to NDC
            // z [0, 1]. Light-space z is negative in front of the eye; the
            // nearest corner has the *largest* z (least negative), so
            // ortho-near = -max.z and ortho-far = -min.z.
            let ortho = Mat4::orthographic_rh(min.x, max.x, min.y, max.y, -max.z, -min.z);
            ortho * light_view
        })
    }
}

/// Engine-standard camera uniform layout — view-projection matrix, the
/// camera's world-space right/up basis vectors, and the camera's world
/// position. The basis fields are present so the same buffer can serve both
/// world-space mesh pipelines and camera-facing billboard pipelines without
/// re-binding; the position is here so lit materials can compute a correct
/// view direction per fragment. Shaders that don't need a given field can
/// declare a smaller WGSL struct (e.g. just `view_proj`) and ignore the
/// trailing bytes — order is stable.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformData {
    pub view_proj: Mat4,
    /// `xyz` = world-space right, `w` = padding.
    pub right: Vec4,
    /// `xyz` = world-space up, `w` = padding.
    pub up: Vec4,
    /// `xyz` = camera world position, `w` = padding.
    pub position: Vec4,
}

impl CameraUniformData {
    pub fn from_camera(camera: &Camera) -> Self {
        let (right, up) = camera.billboard_basis();
        Self {
            view_proj: camera.view_proj(),
            right: right.extend(0.0),
            up: up.extend(0.0),
            position: camera.position.extend(0.0),
        }
    }
}

/// Owns the GPU resources behind the engine-standard camera uniform: the
/// uniform buffer, its bind group layout (binding 0, vertex-stage uniform),
/// and a bind group. Construct once at view init, call [`write`](Self::write)
/// each frame with the latest [`Camera`], and pass [`layout`](Self::layout)
/// when building any pipeline that reads the camera.
///
/// ```text
/// init:    let camera_binding = CameraBinding::new(&renderer.device);
///          let layout = device.create_pipeline_layout(... bind_group_layouts: &[Some(camera_binding.layout())] ...);
/// frame:   camera_binding.write(&renderer.queue, &self.camera);
///          pass.set_bind_group(0, camera_binding.bind_group(), &[]);
/// ```
pub struct CameraBinding {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl CameraBinding {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("currawong camera uniform"),
            size: std::mem::size_of::<CameraUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("currawong camera bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // VERTEX_FRAGMENT, not just VERTEX: lit materials read the
                // camera position in the fragment stage to compute the view
                // direction per fragment.
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("currawong camera bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    /// Upload the latest camera state. Computes the billboard basis vectors
    /// internally so a single uniform buffer serves both mesh and billboard
    /// pipelines.
    pub fn write(&self, queue: &wgpu::Queue, camera: &Camera) {
        let data = CameraUniformData::from_camera(camera);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }

    /// The bind group layout. Pass to [`PipelineLayoutDescriptor`](wgpu::PipelineLayoutDescriptor)
    /// when building any pipeline that reads the camera.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// The bind group, for `pass.set_bind_group(N, ..., &[])` each frame.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
        Camera {
            position: Vec3::new(10.0, -10.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Z,
            fov_y_radians: 60_f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
            zone: None,
        }
    }

    fn slice_corners_world(camera: &Camera, near_d: f32, far_d: f32) -> [Vec3; 8] {
        let view_inv = camera.view_matrix().inverse();
        let tan_half_fov = (camera.fov_y_radians * 0.5).tan();
        let half_h_n = near_d * tan_half_fov;
        let half_w_n = half_h_n * camera.aspect;
        let half_h_f = far_d * tan_half_fov;
        let half_w_f = half_h_f * camera.aspect;
        let corners_view = [
            Vec3::new(-half_w_n, -half_h_n, -near_d),
            Vec3::new(half_w_n, -half_h_n, -near_d),
            Vec3::new(-half_w_n, half_h_n, -near_d),
            Vec3::new(half_w_n, half_h_n, -near_d),
            Vec3::new(-half_w_f, -half_h_f, -far_d),
            Vec3::new(half_w_f, -half_h_f, -far_d),
            Vec3::new(-half_w_f, half_h_f, -far_d),
            Vec3::new(half_w_f, half_h_f, -far_d),
        ];
        std::array::from_fn(|i| view_inv.transform_point3(corners_view[i]))
    }

    #[test]
    fn cascade_split_distances_endpoints_and_monotonic() {
        let camera = test_camera();
        let splits = camera.cascade_split_distances(0.75);
        assert!((splits[3] - camera.far).abs() < 1e-3);
        for i in 1..4 {
            assert!(
                splits[i] > splits[i - 1],
                "splits not monotonically increasing: {:?}",
                splits
            );
        }
        for s in splits {
            assert!(s >= camera.near - 1e-3 && s <= camera.far + 1e-3);
        }
    }

    #[test]
    fn cascade_splits_uniform_at_lambda_zero() {
        let camera = test_camera();
        let splits = camera.cascade_split_distances(0.0);
        // At lambda=0 splits are pure uniform spacing.
        let step = (camera.far - camera.near) / 4.0;
        for (i, s) in splits.iter().enumerate() {
            let expected = camera.near + step * (i + 1) as f32;
            assert!((s - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn fit_shadow_cascades_contains_frustum_corners() {
        let camera = test_camera();
        let sun = Vec3::new(0.3, -0.2, 0.9).normalize();
        let splits = camera.cascade_split_distances(0.75);
        let mats = camera.fit_shadow_cascades(sun, splits);

        for i in 0..4 {
            let near_d = if i == 0 { camera.near } else { splits[i - 1] };
            let far_d = splits[i];
            let corners_world = slice_corners_world(&camera, near_d, far_d);

            for c_world in corners_world {
                let clip = mats[i] * c_world.extend(1.0);
                let ndc = clip.truncate() / clip.w;
                let eps = 1e-3;
                assert!(
                    ndc.x >= -1.0 - eps && ndc.x <= 1.0 + eps,
                    "cascade {} ndc.x = {}",
                    i,
                    ndc.x
                );
                assert!(
                    ndc.y >= -1.0 - eps && ndc.y <= 1.0 + eps,
                    "cascade {} ndc.y = {}",
                    i,
                    ndc.y
                );
                assert!(
                    ndc.z >= -eps && ndc.z <= 1.0 + eps,
                    "cascade {} ndc.z = {}",
                    i,
                    ndc.z
                );
            }
        }
    }

    #[test]
    fn fit_shadow_cascades_handles_vertical_sun() {
        // Sun straight overhead — exercises the up-vector fallback branch.
        let camera = test_camera();
        let sun = Vec3::Z;
        let splits = camera.cascade_split_distances(0.75);
        let mats = camera.fit_shadow_cascades(sun, splits);
        for (i, m) in mats.iter().enumerate() {
            assert!(m.is_finite(), "cascade {} matrix not finite", i);
        }
    }
}
