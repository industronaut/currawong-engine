//! Mouse picking: screen-space cursor → world ray → terrain hit → grid cell.
//!
//! Picking has two halves. The first half — pixel-space cursor to a
//! world-space [`Ray`] — is a pure projection inverse and belongs on
//! [`Camera`]: any future picking (units, props, gizmos) reuses it. The
//! second half — ray against a tile grid, producing a hovered cell — is
//! terrain-specific and lives on [`TerrainPicker`].
//!
//! [`TerrainPicker`] is a value-type View helper in the same family as
//! [`Camera`] and [`OrbitRig`](super::OrbitRig). A View holds one, forwards
//! [`WindowEvent`]s to [`handle_event`](TerrainPicker::handle_event) from
//! `input`, calls [`update`](TerrainPicker::update) from `update`, and
//! reads [`hover`](TerrainPicker::hover) from `render` / `ui`.
//!
//! The picker keeps two parallel hover results:
//!
//! - `latest_plane_hover` — zero-latency ray vs. the Z = 0 plane, exact on
//!   flat terrain and wrong on slopes. Always runs.
//! - `latest_id_hover` — derived from the engine's GPU hit-ID buffer
//!   ([`Renderer::hit_id_hover`](super::Renderer::hit_id_hover)). Correct
//!   on sloped terrain and at cliff edges, but 1–3 frames behind because
//!   readback is asynchronous. Set each frame via [`Self::set_id_hover`].
//!
//! [`Self::hover`] prefers the ID-buffer result when present and falls back
//! to the plane ray otherwise, so the camera-orbit feel stays snappy at the
//! moment readback isn't yet caught up while still being correct everywhere
//! else.

use glam::{IVec2, Vec2, Vec3, Vec4};
use winit::event::WindowEvent;

use crate::sim::Grid;

use super::camera::Camera;
use super::picking_buffer::HitTarget;

/// A world-space ray with a normalised direction. Picking primitives —
/// `intersect_z_plane`, future `intersect_aabb`, etc. — hang off this type.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Point along the ray at parameter `t`. `t = 0` is the origin; positive
    /// `t` walks forward along `direction`.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Intersect with the horizontal plane `z = plane_z`. Returns `None` if
    /// the ray is parallel to the plane (no hit) or hits behind the origin
    /// (cursor is above the horizon — the ray would have to walk backwards
    /// to reach the ground).
    pub fn intersect_z_plane(&self, plane_z: f32) -> Option<Vec3> {
        // Parallel: direction.z ≈ 0 → infinite or no solutions; treat as miss.
        if self.direction.z.abs() < 1e-6 {
            return None;
        }
        let t = (plane_z - self.origin.z) / self.direction.z;
        if t < 0.0 {
            return None;
        }
        Some(self.at(t))
    }
}

impl Camera {
    /// Unproject a cursor position (in physical pixels, top-left origin) to
    /// a world-space ray. `viewport_px` is the full window size in physical
    /// pixels — `Renderer::window.inner_size()` is the canonical source.
    ///
    /// Ray origin is the near-plane point under the cursor; direction is
    /// normalised toward the far-plane point. This matches wgpu/Vulkan
    /// clip-space depth conventions (`z ∈ [0, 1]`); don't reuse for an
    /// OpenGL-NDC pipeline without flipping `z`.
    pub fn ray_from_cursor(&self, cursor_px: Vec2, viewport_px: Vec2) -> Ray {
        // Pixel → NDC. Pixel Y grows downward; NDC Y grows upward — hence
        // the flip on the second component.
        let ndc = Vec2::new(
            (cursor_px.x / viewport_px.x.max(1.0)) * 2.0 - 1.0,
            1.0 - (cursor_px.y / viewport_px.y.max(1.0)) * 2.0,
        );
        let inv_vp = self.view_proj().inverse();
        let near_h = inv_vp * Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
        let far_h = inv_vp * Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
        let near = near_h.truncate() / near_h.w;
        let far = far_h.truncate() / far_h.w;
        Ray {
            origin: near,
            direction: (far - near).normalize(),
        }
    }
}

/// A single hovered-cell result: the cell's grid coordinate and the precise
/// world position the picking ray hit. The world position is useful for
/// drawing a cursor in world space without re-running the math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hover {
    pub cell: IVec2,
    pub world_pos: Vec3,
}

/// View-side mouse-over picker for tile-grid terrain. Holds the last cursor
/// position and the most recent picking result; the View owns one, feeds it
/// window events, and reads `hover()` when drawing.
///
/// Parameterised on the same [`Grid`] type the zone's terrain is built on,
/// so square and hex grids drop in without changing the picker. The
/// `tile_size` field mirrors the mesher's `tile_size` — picker scales the
/// ray hit into the grid's canonical unit space before asking
/// [`Grid::cell_at`] which cell the cursor is over.
pub struct TerrainPicker<G: Grid> {
    grid: G,
    tile_size: f32,
    plane_z: f32,
    last_cursor_px: Option<Vec2>,
    plane_hover: Option<Hover>,
    id_hover: Option<Hover>,
}

impl<G: Grid> TerrainPicker<G> {
    /// Build a picker for a terrain on `grid` with the given mesher
    /// `tile_size`. The ground plane defaults to `z = 0`; use
    /// [`with_plane_z`](Self::with_plane_z) to override.
    pub fn new(grid: G, tile_size: f32) -> Self {
        Self {
            grid,
            tile_size,
            plane_z: 0.0,
            last_cursor_px: None,
            plane_hover: None,
            id_hover: None,
        }
    }

    /// Override the picking plane height. Useful if the zone's base ground
    /// sits at a non-zero Z (multi-floor buildings, sunken arenas).
    pub fn with_plane_z(mut self, z: f32) -> Self {
        self.plane_z = z;
        self
    }

    /// Ingest a winit window event. Tracks cursor position from
    /// `CursorMoved` and clears both hover results on `CursorLeft`. Events
    /// the picker doesn't care about are ignored.
    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.last_cursor_px = Some(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => {
                self.last_cursor_px = None;
                self.plane_hover = None;
                self.id_hover = None;
            }
            _ => {}
        }
    }

    /// Re-run the ray-vs-plane pick. Call once per frame from
    /// [`View::update`](crate::View::update); the pick has to re-run every
    /// frame even when the cursor is stationary because the camera might
    /// have moved. Updates the plane-hover slot only; ID-buffer hover is
    /// fed in via [`Self::set_id_hover`].
    pub fn update(&mut self, camera: &Camera, viewport_px: Vec2) {
        let Some(cursor) = self.last_cursor_px else {
            self.plane_hover = None;
            return;
        };
        let ray = camera.ray_from_cursor(cursor, viewport_px);
        let Some(hit) = ray.intersect_z_plane(self.plane_z) else {
            self.plane_hover = None;
            return;
        };
        let canonical_xy = Vec2::new(hit.x, hit.y) / self.tile_size;
        let cell = self.grid.cell_at(canonical_xy);
        self.plane_hover = Some(Hover {
            cell,
            world_pos: hit,
        });
    }

    /// Feed the engine's latest hit-ID result into the picker. The View's
    /// per-frame update typically does
    /// `picker.set_id_hover(renderer.hit_id_hover())`. Only
    /// [`HitTarget::TerrainCell`] is consumed (mesh-object hits go through
    /// a different path); other variants clear the slot.
    ///
    /// `world_pos` on the resulting [`Hover`] is the cell centre on the
    /// picking plane — the ID buffer gives us a cell, not a precise hit
    /// point. Callers that need an exact world position should consult
    /// [`Self::plane_hover`] for the ray result instead.
    pub fn set_id_hover(&mut self, target: Option<HitTarget>) {
        self.id_hover = target.map(|HitTarget::TerrainCell { cell, .. }| {
            let centre_canonical: Vec2 = (0..G::CORNERS_PER_CELL)
                .map(|i| self.grid.corner_xy(cell, i))
                .fold(Vec2::ZERO, |acc, v| acc + v)
                / G::CORNERS_PER_CELL as f32;
            let centre = centre_canonical * self.tile_size;
            Hover {
                cell,
                world_pos: Vec3::new(centre.x, centre.y, self.plane_z),
            }
        });
    }

    /// Latest pick result. Prefers the ID-buffer hover when present
    /// (correct on slopes / at cliff edges, 1–3 frames of latency) and
    /// falls back to the plane ray (zero latency, exact only on flat
    /// ground). Returns `None` if both are clear (cursor off-window or
    /// looking at the sky with no readback yet).
    pub fn hover(&self) -> Option<Hover> {
        self.id_hover.or(self.plane_hover)
    }

    /// The ray-vs-plane hover specifically — useful when the caller needs
    /// the exact world hit point regardless of latency, e.g. positioning a
    /// world-space cursor over flat terrain.
    pub fn plane_hover(&self) -> Option<Hover> {
        self.plane_hover
    }

    /// The ID-buffer hover specifically — useful when the caller wants to
    /// distinguish "the GPU saw this cell under the cursor" from a plane
    /// fallback.
    pub fn id_hover(&self) -> Option<Hover> {
        self.id_hover
    }

    /// Cursor's last known pixel position. `None` between `CursorLeft` and
    /// the next `CursorMoved`. Exposed mainly for views that want to draw
    /// a screen-space crosshair or feed picking to a second consumer.
    pub fn last_cursor_px(&self) -> Option<Vec2> {
        self.last_cursor_px
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::{HexGrid, SquareGrid};

    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn looking_down_camera() -> Camera {
        // Camera directly above the origin, looking straight down +Z → 0,
        // up vector along +Y so screen +X is world +X and screen up is +Y.
        // Square viewport keeps aspect math out of the picture.
        Camera {
            position: Vec3::new(0.0, 0.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60_f32.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            zone: None,
        }
    }

    #[test]
    fn ray_through_centre_hits_target() {
        // Cursor at the centre of the viewport projects to a ray that
        // passes through the camera's look-at target — the most basic
        // sanity check on the unproject.
        let cam = looking_down_camera();
        let ray = cam.ray_from_cursor(Vec2::new(400.0, 400.0), Vec2::new(800.0, 800.0));
        let hit = ray.intersect_z_plane(0.0).expect("looking down hits z=0");
        assert!(approx_eq(hit.x, 0.0, 1e-4));
        assert!(approx_eq(hit.y, 0.0, 1e-4));
        assert!(approx_eq(hit.z, 0.0, 1e-4));
    }

    #[test]
    fn ray_right_of_centre_hits_positive_x() {
        // Cursor right of centre → ray hits +X side of the target. Pins
        // the NDC X mapping: pixel +X must be world +X here (camera up is
        // +Y, looking down −Z).
        let cam = looking_down_camera();
        let ray = cam.ray_from_cursor(Vec2::new(600.0, 400.0), Vec2::new(800.0, 800.0));
        let hit = ray.intersect_z_plane(0.0).expect("looking down hits z=0");
        assert!(hit.x > 0.0, "hit {hit:?} should have +x");
        assert!(approx_eq(hit.y, 0.0, 1e-4));
    }

    #[test]
    fn ray_above_centre_hits_positive_y() {
        // Cursor above centre (smaller pixel Y) → ray hits +Y. Pins the
        // pixel→NDC Y flip: pixel +Y goes down on screen, world +Y goes up
        // in the scene under this camera.
        let cam = looking_down_camera();
        let ray = cam.ray_from_cursor(Vec2::new(400.0, 200.0), Vec2::new(800.0, 800.0));
        let hit = ray.intersect_z_plane(0.0).expect("looking down hits z=0");
        assert!(approx_eq(hit.x, 0.0, 1e-4));
        assert!(hit.y > 0.0, "hit {hit:?} should have +y");
    }

    #[test]
    fn intersect_z_plane_misses_when_above_horizon() {
        // Ray pointing up never hits the ground plane below it.
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 1.0),
            direction: Vec3::new(0.0, 0.5, 0.5).normalize(),
        };
        assert_eq!(ray.intersect_z_plane(0.0), None);
    }

    #[test]
    fn intersect_z_plane_misses_when_parallel() {
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 1.0),
            direction: Vec3::new(1.0, 0.0, 0.0),
        };
        assert_eq!(ray.intersect_z_plane(0.0), None);
    }

    #[test]
    fn picker_reports_cell_under_cursor_square() {
        // Top-down camera, square grid, tile_size = 1. Cursor at the
        // viewport centre picks the cell containing (0, 0) on a
        // SquareGrid: cell (0, 0) since 0.0 maps under floor semantics.
        let cam = looking_down_camera();
        let mut picker = TerrainPicker::new(SquareGrid, 1.0);
        // Simulate a CursorMoved event the View would forward.
        picker.last_cursor_px = Some(Vec2::new(400.0, 400.0));
        picker.update(&cam, Vec2::new(800.0, 800.0));
        let hover = picker.hover().expect("centred cursor must hit ground");
        assert_eq!(hover.cell, IVec2::new(0, 0));
        assert!(approx_eq(hover.world_pos.x, 0.0, 1e-4));
        assert!(approx_eq(hover.world_pos.y, 0.0, 1e-4));
    }

    #[test]
    fn picker_clears_hover_when_cursor_leaves() {
        let cam = looking_down_camera();
        let mut picker = TerrainPicker::new(SquareGrid, 1.0);
        picker.last_cursor_px = Some(Vec2::new(400.0, 400.0));
        picker.update(&cam, Vec2::new(800.0, 800.0));
        assert!(picker.hover().is_some());

        // CursorLeft must drop both the cached position and the hover.
        picker.handle_event(&WindowEvent::CursorLeft {
            device_id: winit::event::DeviceId::dummy(),
        });
        assert!(picker.last_cursor_px().is_none());
        assert!(picker.hover().is_none());
    }

    #[test]
    fn picker_clears_hover_when_ray_misses_plane() {
        // Camera above the plane looking *up* — ray walks away from z=0,
        // not toward it. Picker must drop the hover rather than report a
        // stale one from a prior frame.
        let cam = Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::new(0.0, 0.0, 20.0),
            up: Vec3::Y,
            ..looking_down_camera()
        };
        let mut picker = TerrainPicker::new(SquareGrid, 1.0);
        picker.last_cursor_px = Some(Vec2::new(400.0, 400.0));
        picker.update(&cam, Vec2::new(800.0, 800.0));
        assert!(picker.hover().is_none());
    }

    #[test]
    fn picker_honours_tile_size_for_square_grid() {
        // Same world hit, twice the tile_size → half the cell index in each
        // axis. Cursor at the centre of the viewport hits world (0,0), so
        // exercise a non-zero cell by nudging the camera off-origin.
        let cam = Camera {
            position: Vec3::new(2.5, 2.5, 10.0),
            target: Vec3::new(2.5, 2.5, 0.0),
            up: Vec3::Y,
            ..looking_down_camera()
        };
        let mut p1 = TerrainPicker::new(SquareGrid, 1.0);
        let mut p2 = TerrainPicker::new(SquareGrid, 2.0);
        for p in [&mut p1, &mut p2] {
            p.last_cursor_px = Some(Vec2::new(400.0, 400.0));
            p.update(&cam, Vec2::new(800.0, 800.0));
        }
        assert_eq!(p1.hover().unwrap().cell, IVec2::new(2, 2));
        assert_eq!(p2.hover().unwrap().cell, IVec2::new(1, 1));
    }

    #[test]
    fn picker_works_with_hex_grid() {
        // Centre of the viewport over the origin → cell (0, 0) on a hex
        // grid as well. Type-check that the picker is parametric and the
        // call site doesn't change.
        let cam = looking_down_camera();
        let mut picker = TerrainPicker::new(HexGrid, 1.0);
        picker.last_cursor_px = Some(Vec2::new(400.0, 400.0));
        picker.update(&cam, Vec2::new(800.0, 800.0));
        let hover = picker.hover().expect("centred cursor must hit ground");
        assert_eq!(hover.cell, IVec2::new(0, 0));
    }
}
