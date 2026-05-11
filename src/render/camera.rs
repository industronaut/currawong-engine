//! Perspective camera helper for views that render in 3D world space.

use glam::{Mat4, Vec3, Vec4};

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
            position: Vec3::new(0.0, -5.0, 2.0),
            target: Vec3::ZERO,
            up: Vec3::Z,
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
}

/// Engine-standard camera uniform layout — view-projection matrix plus the
/// camera's world-space right/up basis vectors. The basis fields are present
/// so the same buffer can serve both world-space mesh pipelines and
/// camera-facing billboard pipelines without re-binding. Shaders that don't
/// need the basis can declare a smaller WGSL struct (e.g. just `view_proj`)
/// and ignore the trailing bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformData {
    pub view_proj: Mat4,
    /// `xyz` = world-space right, `w` = padding.
    pub right: Vec4,
    /// `xyz` = world-space up, `w` = padding.
    pub up: Vec4,
}

impl CameraUniformData {
    pub fn from_camera(camera: &Camera) -> Self {
        let (right, up) = camera.billboard_basis();
        Self {
            view_proj: camera.view_proj(),
            right: right.extend(0.0),
            up: up.extend(0.0),
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
                visibility: wgpu::ShaderStages::VERTEX,
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
