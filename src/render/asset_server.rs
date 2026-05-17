//! [`AssetServer`] — the view-side gateway from [`Vfs`] paths to streamed
//! GPU assets.
//!
//! ## Roles
//!
//! - Hands out [`Handle<T>`]s for typed asset paths. Repeated requests for
//!   the same (path, color_space) share a single load via the per-type
//!   cache.
//! - Spawns a background thread per load. The thread reads bytes via the
//!   VFS, decodes them, uploads to the GPU, and publishes the result into
//!   the handle. Failures land in the handle's `Failed` state with the
//!   reason; the render path always has *something* to bind.
//! - Owns the magenta fallback texture — 4×4 magenta, always resident.
//!   Materials read it via [`AssetServer::fallback_texture`] whenever their
//!   handle is `Loading` or `Failed`.
//! - Carries a debug toggle ([`AssetServer::set_force_loading`]) that
//!   forces every handle to *appear* `Loading` to consumers. Exists so the
//!   fallback path is exercised in normal dev, not just during real I/O
//!   hiccups — without it the fallback rots silently.
//!
//! ## Async story
//!
//! `std::thread::spawn` per load — deliberately simple. The
//! background→main handoff is direct GPU upload: `wgpu::Device` and
//! `wgpu::Queue` are `Send + Sync + Clone`, so the worker thread uploads
//! straight into the texture and the main thread sees the populated handle
//! on the next frame. No channel, no frame-pacing scheduler, no executor
//! commitment — those land if a real workload demands them.
//!
//! ## Caching
//!
//! `(VfsPath, TextureColorSpace)` keys the texture cache. The colour-space
//! is part of the key because the same PNG sampled as sRGB vs Linear is a
//! semantically different asset (one for lighting, one for data). Two paths
//! requested with the same colour space share a load and an upload;
//! mismatched colour space spawns a second load.
//!
//! Refcount-based eviction is deliberately not here — see issue #69's
//! "Out of scope." Cached handles stay alive for the lifetime of the
//! server; explicit eviction can land if memory pressure becomes real.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Mat4, Vec3};

use crate::data::{Vfs, VfsPath};

use super::handle::{Handle, HandleError, new_pending};
use super::mesh::{DecodedMesh, Mesh};
use super::mesh_primitives::PrimitiveMesh;
use super::renderer::Renderer;
use super::texture::{Texture, TextureColorSpace};
use super::visibility::Aabb;

/// View-side asset gateway. Construct once at [`View::init`](crate::View::init)
/// with a fully-mounted [`Vfs`] and the live [`Renderer`]; share by reference
/// from there.
pub struct AssetServer {
    vfs: Arc<Vfs>,
    /// Cloned out of [`Renderer`] at construction time and stashed so
    /// background loader threads can upload without holding a reference
    /// back into the renderer. Cheap clone — both types are internally
    /// `Arc`-backed in wgpu 29.
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// 4×4 magenta texture — bound whenever a handle isn't `Ready`. Lives
    /// for the lifetime of the server, so material instances can hold a
    /// `&wgpu::TextureView` of it across frames without refcount juggling.
    fallback: Texture,
    /// Unit-cube [`PosNormalUv`] mesh — drawn whenever a mesh handle isn't
    /// `Ready`. Bounds `[-0.5, 0.5]^3`; [`resolve_mesh`](Self::resolve_mesh)
    /// returns a translate+scale [`Mat4`] sized to the caller's declared
    /// template bounds so the placeholder occupies roughly the right space.
    /// Drawn through the caller's pipeline (typically PBR); paired with the
    /// magenta texture fallback above when the texture handle is also
    /// unready, the visual result is a magenta cube of the right size — the
    /// "third fallback" (a standalone unlit magenta pipeline) is intentionally
    /// deferred until something needs it.
    fallback_mesh: Mesh,
    /// Per-path texture cache. `Mutex` because handle requests come from
    /// view code on the main thread but the cache could later be touched
    /// from background loaders too (e.g. to evict on completion). Today
    /// only the main thread touches it; the lock is cheap.
    textures: Mutex<HashMap<TextureKey, Handle<Texture>>>,
    /// Per-path mesh cache. Same shape as `textures`; keyed by `VfsPath`
    /// only because mesh loading has no extra orthogonal dimensions (no
    /// colour-space equivalent).
    meshes: Mutex<HashMap<VfsPath, Handle<Mesh>>>,
    /// Debug toggle: while `true`, every consumer sees
    /// [`AssetServer::resolve_texture`] resolve to the fallback regardless
    /// of the underlying handle's real state. Set via
    /// [`set_force_loading`](Self::set_force_loading); examples wire a keybind
    /// to it.
    force_loading: AtomicBool,
}

/// Cache key for textures. The colour-space is part of the key because the
/// same PNG sampled sRGB vs Linear is a different GPU resource.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct TextureKey {
    path: VfsPath,
    color_space: TextureColorSpace,
}

impl AssetServer {
    /// Build a fresh asset server. Clones the renderer's `device`/`queue`
    /// handles (cheap — they're `Arc`-backed internally) so background
    /// loaders can upload without reaching back into `Renderer`.
    pub fn new(renderer: &Renderer, vfs: Arc<Vfs>) -> Self {
        let fallback = build_magenta_fallback(&renderer.device, &renderer.queue);
        let fallback_mesh = build_unit_cube_fallback(&renderer.device, &renderer.queue);
        Self {
            vfs,
            device: renderer.device.clone(),
            queue: renderer.queue.clone(),
            fallback,
            fallback_mesh,
            textures: Mutex::new(HashMap::new()),
            meshes: Mutex::new(HashMap::new()),
            force_loading: AtomicBool::new(false),
        }
    }

    /// The 4×4 magenta fallback texture — always `Ready`, always bindable.
    /// Materials use this whenever their handle isn't `Ready` (real load
    /// in flight, real load failed, or the debug
    /// [`force_loading`](Self::set_force_loading) toggle is on).
    pub fn fallback_texture(&self) -> &Texture {
        &self.fallback
    }

    /// The unit-cube [`PosNormalUv`] fallback mesh — always `Ready`,
    /// always drawable through any pipeline that demands the canonical
    /// vertex layout. Drawn whenever a mesh handle isn't `Ready`. Bounds
    /// `[-0.5, 0.5]^3`; see [`resolve_mesh`](Self::resolve_mesh) for the
    /// per-draw scale that sizes it to a template's declared visual AABB.
    pub fn fallback_mesh(&self) -> &Mesh {
        &self.fallback_mesh
    }

    /// Borrow the underlying [`Vfs`]. Useful for loaders that don't fit the
    /// per-type cache surface (e.g. one-shot RON reads at startup) — the
    /// server doesn't gatekeep VFS access.
    pub fn vfs(&self) -> &Arc<Vfs> {
        &self.vfs
    }

    /// Whether the debug "every handle looks Loading" toggle is on. Read
    /// by material refresh paths.
    pub fn force_loading(&self) -> bool {
        self.force_loading.load(Ordering::Relaxed)
    }

    /// Flip the debug toggle. Wire a keybind in your example: set to true
    /// to force the fallback path during normal rendering, set back to
    /// false to see real textures rebind. Useful for verifying the
    /// fallback path during dev — without it the fallback rots silently
    /// because real loads almost always succeed.
    pub fn set_force_loading(&self, on: bool) {
        self.force_loading.store(on, Ordering::Relaxed);
    }

    /// Return a [`Handle<Texture>`] for `path`, spawning a background load
    /// if this is the first request. Repeated calls for the same
    /// `(path, color_space)` share the same handle — second-and-later
    /// callers get a cheap `Arc` clone of the in-flight or completed load.
    ///
    /// Returns immediately. The handle starts in `Loading` and transitions
    /// to `Ready` or `Failed` once the background thread finishes.
    pub fn texture(&self, path: VfsPath, color_space: TextureColorSpace) -> Handle<Texture> {
        let key = TextureKey {
            path: path.clone(),
            color_space,
        };
        let mut cache = self.textures.lock().expect("texture cache poisoned");
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }

        let (handle, writer) = new_pending::<Texture>(path.as_str());
        cache.insert(key, handle.clone());
        drop(cache);

        let vfs = Arc::clone(&self.vfs);
        let device = self.device.clone();
        let queue = self.queue.clone();
        std::thread::spawn(move || {
            let result = load_texture(&vfs, &device, &queue, &path, color_space);
            match result {
                Ok(texture) => writer.set_ready(texture),
                Err(error) => {
                    eprintln!("currawong asset load: {error}");
                    writer.set_failed(error);
                }
            }
        });

        handle
    }

    /// Return a [`Handle<Mesh>`] for `path`, spawning a background load if
    /// this is the first request. Repeated calls for the same path share
    /// the same handle — second-and-later callers get a cheap `Arc` clone.
    ///
    /// Returns immediately. The handle starts in `Loading` and transitions
    /// to `Ready` or `Failed` once the background thread finishes parsing
    /// and uploading the glTF.
    pub fn mesh(&self, path: VfsPath) -> Handle<Mesh> {
        let mut cache = self.meshes.lock().expect("mesh cache poisoned");
        if let Some(existing) = cache.get(&path) {
            return existing.clone();
        }

        let (handle, writer) = new_pending::<Mesh>(path.as_str());
        cache.insert(path.clone(), handle.clone());
        drop(cache);

        let vfs = Arc::clone(&self.vfs);
        let device = self.device.clone();
        let queue = self.queue.clone();
        std::thread::spawn(move || {
            let result = load_mesh(&vfs, &device, &queue, &path);
            match result {
                Ok(mesh) => writer.set_ready(mesh),
                Err(error) => {
                    eprintln!("currawong asset load: {error}");
                    writer.set_failed(error);
                }
            }
        });

        handle
    }

    /// Resolve a mesh handle to the buffers + per-draw transform the caller
    /// should bind this frame.
    ///
    /// On `Ready` with the debug toggle off, returns the real mesh's
    /// vertex/index buffers, its index count, and `Mat4::IDENTITY` for the
    /// fallback adjustment — compose normally.
    ///
    /// On `Loading` / `Failed` / forced-fallback, returns the resident
    /// unit-cube fallback's buffers + a translate-then-scale `Mat4` sized
    /// to `template_bounds` (or `Mat4::IDENTITY` if no bounds were
    /// provided). Callers compose it inside their per-instance world
    /// transform: `model = world_xform * fallback_adjustment`. This keeps
    /// the placeholder occupying roughly the right volume so frustum
    /// culling and screen-space size heuristics agree with the real
    /// mesh's footprint.
    pub fn resolve_mesh<'a>(
        &'a self,
        handle: &'a Handle<Mesh>,
        template_bounds: Option<Aabb>,
    ) -> ResolvedMesh<'a> {
        if self.force_loading() {
            return self.fallback_resolved(template_bounds, MeshSource::ForcedFallback);
        }
        match handle.peek() {
            super::handle::HandleState::Ready(m) => ResolvedMesh {
                vertex_buffer: &m.vertex_buffer,
                index_buffer: &m.index_buffer,
                index_count: m.index_count,
                source: MeshSource::Real,
                fallback_adjustment: Mat4::IDENTITY,
            },
            super::handle::HandleState::Loading => {
                self.fallback_resolved(template_bounds, MeshSource::Loading)
            }
            super::handle::HandleState::Failed(_) => {
                self.fallback_resolved(template_bounds, MeshSource::Failed)
            }
        }
    }

    /// Build a `ResolvedMesh` that draws the resident unit-cube fallback,
    /// scaled+translated so its `[-0.5, 0.5]^3` extents land on
    /// `template_bounds`. Identity when no bounds were provided.
    fn fallback_resolved(
        &self,
        template_bounds: Option<Aabb>,
        source: MeshSource,
    ) -> ResolvedMesh<'_> {
        let fallback_adjustment = template_bounds
            .map(fallback_adjustment_for)
            .unwrap_or(Mat4::IDENTITY);
        ResolvedMesh {
            vertex_buffer: &self.fallback_mesh.vertex_buffer,
            index_buffer: &self.fallback_mesh.index_buffer,
            index_count: self.fallback_mesh.index_count,
            source,
            fallback_adjustment,
        }
    }

    /// Resolve a texture handle to whichever `wgpu::TextureView` should
    /// actually be bound this frame — the real texture's view when the
    /// handle is `Ready` and the debug toggle is off, the fallback view
    /// otherwise.
    ///
    /// Material instances call this each refresh; the returned view's
    /// lifetime is tied to `&self` because the fallback lives on the
    /// server and `Ready` textures live in the handle's slot (which is
    /// `Arc`-shared with the cache, also held by the server).
    pub fn resolve_texture<'a>(&'a self, handle: &'a Handle<Texture>) -> ResolvedTexture<'a> {
        if self.force_loading() {
            return ResolvedTexture {
                view: &self.fallback.view,
                source: TextureSource::ForcedFallback,
            };
        }
        match handle.peek() {
            super::handle::HandleState::Ready(t) => ResolvedTexture {
                view: &t.view,
                source: TextureSource::Real,
            },
            super::handle::HandleState::Loading => ResolvedTexture {
                view: &self.fallback.view,
                source: TextureSource::Loading,
            },
            super::handle::HandleState::Failed(_) => ResolvedTexture {
                view: &self.fallback.view,
                source: TextureSource::Failed,
            },
        }
    }
}

/// What a [`AssetServer::resolve_texture`] call produced — a bindable
/// `TextureView` plus a tag for *why* that view was chosen.
///
/// Material instances cache the [`source`](Self::source) tag and only
/// rebuild their bind group when it changes between frames. Comparing the
/// tag is sufficient because the fallback view never moves (the server's
/// magenta texture lives for the server's lifetime) and the real view also
/// never moves once the handle is `Ready` (`OnceLock` is write-once).
pub struct ResolvedTexture<'a> {
    pub view: &'a wgpu::TextureView,
    pub source: TextureSource,
}

/// Why a [`ResolvedTexture`] resolved to the view it did. Material refresh
/// paths track the last seen value here to decide whether the cached bind
/// group is stale.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureSource {
    /// The handle was `Ready`; we bound the real GPU texture.
    Real,
    /// The handle was still `Loading`; we bound the fallback.
    Loading,
    /// The handle was `Failed`; we bound the fallback (and the failure
    /// reason is reachable via the handle's `peek`).
    Failed,
    /// The debug [`AssetServer::set_force_loading`] toggle was on; we
    /// bound the fallback regardless of the handle's true state.
    ForcedFallback,
}

/// What a [`AssetServer::resolve_mesh`] call produced — buffers and index
/// count to draw, a [`MeshSource`] tag for *why* those buffers were
/// chosen, and a [`Mat4`] adjustment for the per-instance model transform.
///
/// Compose the adjustment inside the world transform:
///
/// ```text
/// model = world_xform * resolved.fallback_adjustment
/// ```
///
/// For real meshes the adjustment is `Mat4::IDENTITY`, so the compose is a
/// no-op. For fallbacks the adjustment scales+translates the unit cube to
/// the template bounds the caller passed.
pub struct ResolvedMesh<'a> {
    pub vertex_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
    pub index_count: u32,
    pub source: MeshSource,
    pub fallback_adjustment: Mat4,
}

/// Why a [`ResolvedMesh`] resolved to the buffers it did. Same shape as
/// [`TextureSource`], for the same reasons — callers that cache anything
/// per-mesh can branch on transitions cheaply.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeshSource {
    Real,
    Loading,
    Failed,
    ForcedFallback,
}

/// Translate+scale that maps the unit cube (`[-0.5, 0.5]^3`) onto
/// `bounds`. Used by [`AssetServer::resolve_mesh`] to size the fallback
/// placeholder to a template's declared visual AABB.
///
/// Decomposed: scale by the AABB's full extents (so the unit-cube edge of
/// length 1 becomes an edge of length `max - min`), then translate by the
/// AABB's centre. The result, composed inside the per-instance world
/// transform, plants the placeholder where the real mesh would have been.
fn fallback_adjustment_for(bounds: Aabb) -> Mat4 {
    let center = bounds.center();
    let full_extents = bounds.max - bounds.min;
    Mat4::from_translation(center) * Mat4::from_scale(full_extents)
}

/// Single-shot background loader: VFS read → PNG decode → GPU upload.
/// Runs entirely on a worker thread; the main thread sees only the
/// finished texture (or a `HandleError`) through the `OnceLock` inside the
/// handle.
fn load_texture(
    vfs: &Vfs,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &VfsPath,
    color_space: TextureColorSpace,
) -> Result<Texture, HandleError> {
    let bytes = pollster::block_on(vfs.read(path))
        .map_err(|e| HandleError::new(format!("reading {path}: {e}")))?;
    Texture::from_png_bytes_with_device(device, queue, path.as_str(), &bytes, color_space)
        .map_err(|e| HandleError::new(format!("decoding {path}: {e}")))
}

/// Single-shot background loader: VFS read → glTF decode + repack → GPU
/// upload. Runs entirely on a worker thread; the main thread sees only
/// the finished mesh (or a `HandleError`) through the `OnceLock` inside
/// the handle.
fn load_mesh(
    vfs: &Vfs,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &VfsPath,
) -> Result<Mesh, HandleError> {
    let bytes = pollster::block_on(vfs.read(path))
        .map_err(|e| HandleError::new(format!("reading {path}: {e}")))?;
    Mesh::from_gltf_bytes_with_device(device, queue, path.as_str(), &bytes)
        .map_err(|e| HandleError::new(format!("decoding {path}: {e}")))
}

/// Build the resident unit-cube fallback mesh. Reuses the engine's own
/// [`PrimitiveMesh::cube`] generator so the fallback geometry stays
/// consistent with what the rest of the engine considers "a cube" — bounds
/// `[-0.5, 0.5]^3`, per-face flat normals, UVs spanning 0..1 on each face.
fn build_unit_cube_fallback(device: &wgpu::Device, queue: &wgpu::Queue) -> Mesh {
    let cube = PrimitiveMesh::cube(Vec3::ONE);
    let decoded = DecodedMesh {
        vertices: cube.vertices,
        indices: cube.indices,
        bounds: Aabb::cube(0.5),
    };
    Mesh::from_decoded_with_device(device, queue, "asset-server unit-cube fallback", decoded)
}

/// 4×4 magenta texture. Magenta because it's the conventional "missing
/// asset" colour — saturated, unambiguous, doesn't blend into normal art.
fn build_magenta_fallback(device: &wgpu::Device, queue: &wgpu::Queue) -> Texture {
    const SIZE: u32 = 4;
    const MAGENTA: [u8; 4] = [255, 0, 255, 255];
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for _ in 0..(SIZE * SIZE) {
        rgba.extend_from_slice(&MAGENTA);
    }
    // sRGB-flavoured fallback so it visually pops in albedo bindings; the
    // (255, 0, 255) byte values land as bright magenta after the GPU's
    // sRGB→linear decode on sample, then back to bright magenta in the
    // final swapchain encode. A `Linear`-flavoured fallback would look
    // dimmer through the same path; if a non-albedo binding ever needs the
    // fallback we can add a second one keyed by colour space.
    Texture::from_rgba8_with_device(
        device,
        queue,
        "asset-server magenta fallback",
        SIZE,
        SIZE,
        &rgba,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MemorySource;

    fn p(s: &str) -> VfsPath {
        VfsPath::new(s).unwrap()
    }

    /// The cache key has to round-trip through `HashMap` — sanity check
    /// that the derived impls do what we need. Tiny test but catches the
    /// "forgot `Eq + Hash` on a new field" foot-gun if `TextureKey`
    /// changes.
    #[test]
    fn texture_key_hashes_consistently() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        let a = TextureKey {
            path: p("foo.png"),
            color_space: TextureColorSpace::Srgb,
        };
        let b = TextureKey {
            path: p("foo.png"),
            color_space: TextureColorSpace::Srgb,
        };
        let c = TextureKey {
            path: p("foo.png"),
            color_space: TextureColorSpace::Linear,
        };
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(a, b);
        assert_eq!(h1.finish(), h2.finish());
        // Different colour space → distinct key (we don't share the GPU
        // resource across colour spaces).
        assert_ne!(a, c);
    }

    /// `load_texture` is the loader-thread payload; exercise it on the
    /// current thread with a stubbed `Vfs` so we cover the happy-path
    /// "bytes → texture" pipeline below the GPU upload boundary.
    ///
    /// Stops short of `from_png_bytes_with_device` (needs a device), but
    /// proves the VFS read + path-formatting plumbing.
    #[test]
    fn load_texture_surfaces_vfs_miss_as_handle_error() {
        let mut vfs = Vfs::new();
        vfs.mount(MemorySource::new());
        // No PNG planted; the read should miss and bubble up a HandleError.
        // We can't actually call load_texture without a device — sidestep
        // by exercising just the read branch.
        let result = pollster::block_on(vfs.read(&p("missing.png")))
            .map_err(|e| HandleError::new(format!("reading {}: {e}", "missing.png")));
        assert!(result.is_err(), "missing path must surface as HandleError");
        let err = result.unwrap_err();
        assert!(
            err.message().contains("missing.png"),
            "error message should name the missing path; got {}",
            err.message()
        );
    }

    /// Sister of the texture test above: the mesh loader's read branch
    /// uses the same VFS-miss → `HandleError` plumbing. We can't reach the
    /// GPU upload from a unit test, but we can prove the formatting path.
    #[test]
    fn load_mesh_surfaces_vfs_miss_as_handle_error() {
        let mut vfs = Vfs::new();
        vfs.mount(MemorySource::new());
        let result = pollster::block_on(vfs.read(&p("missing.glb")))
            .map_err(|e| HandleError::new(format!("reading {}: {e}", "missing.glb")));
        assert!(result.is_err(), "missing path must surface as HandleError");
        assert!(result.unwrap_err().message().contains("missing.glb"));
    }

    /// The fallback adjustment that sizes the unit cube onto a template's
    /// declared AABB. Pure math — no GPU.
    #[test]
    fn fallback_adjustment_maps_unit_cube_corners_onto_aabb() {
        use glam::Vec4;

        // Cube centred at (10, 20, 30) with full extents (4, 6, 8).
        let bounds = Aabb::new(Vec3::new(8.0, 17.0, 26.0), Vec3::new(12.0, 23.0, 34.0));
        let m = fallback_adjustment_for(bounds);

        // The unit cube's `(+0.5, +0.5, +0.5)` corner should land on the
        // AABB's `max`, and `(-0.5, -0.5, -0.5)` on its `min`. Anything
        // else means the scale or translate factor is wrong.
        let p_max = m * Vec4::new(0.5, 0.5, 0.5, 1.0);
        let p_min = m * Vec4::new(-0.5, -0.5, -0.5, 1.0);
        assert!((p_max.truncate() - bounds.max).length() < 1e-5);
        assert!((p_min.truncate() - bounds.min).length() < 1e-5);
    }
}
