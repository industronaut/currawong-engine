# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

currawong is a Rust game engine being built from scratch, targeted at simulation-style games (Dwarf Fortress, Factorio, RimWorld). The engine itself is the deliverable — this is an architectural learning project, not a means to ship a specific game. Bias toward building things from scratch when the topic is architecturally interesting; pull in libraries only for orthogonal plumbing (`wgpu`, `winit`, `glam`, `bytemuck`).

## Commands

```bash
cargo build                              # full build (render + sim)
cargo build --no-default-features        # sim-only (no wgpu/winit/pollster)
cargo test                               # all tests
cargo test --no-default-features         # sim tests only — proves sim layer has no render deps
cargo clippy --all-targets
cargo clippy --no-default-features --all-targets

cargo run --example clear                # window with cleared background
cargo run --example triangle             # static colored triangle
cargo run --example input                # input demo + sim speed controls
cargo run --example camera               # sim/view extract + camera demo
cargo run --example terrain              # tile-grid terrain meshing + liquids
cargo run --example trees                # ~200 trees growing under live sim mutation
cargo run --example textured_pbr         # PBR cubes lit by a sim-driven sun
cargo run --example textured_pbr --features egui   # same, with debug overlay
cargo run --example headless             # sim ticking without any window
cargo run --example headless --no-default-features   # proves headless excludes wgpu/winit at compile time
```

A successful build under `--no-default-features` is the primary architectural test — it proves the simulation layer compiles without any rendering dependencies.

## Setup

Activate the in-repo git hooks once per clone:

```bash
git config core.hooksPath .githooks
```

This wires up `.githooks/pre-commit`, which runs `cargo fmt --check` and blocks unformatted commits. Run `cargo fmt` to fix.

## Architecture

The central commitment is **sim/view separation**, modelled on UE-style proxy extraction rather than Unity/Godot scene-graph integration. The codebase splits into two modules with a build-system-enforced boundary:

- `src/sim.rs` + `src/sim/` (submodules `slot_map`, `zone`, `components`, `clock`, `terrain`, `environment`) — sim layer. `sim.rs` is a thin parent that owns the `Simulation` trait and re-exports the submodules' public types so callers see a flat surface. Always compiled. Depends only on `glam` and `std`. Never imports `wgpu` or `winit`.
- `src/render.rs` — view layer. Compiled only with the `render` feature (default on). Owns all GPU + windowing.
- `src/lib.rs` — re-exporter. Conditionally exposes the render layer behind `#[cfg(feature = "render")]`.

The `render` Cargo feature gates `pollster`, `wgpu`, `winit`, `bytemuck`, and `glam/bytemuck`. Render-side examples declare `required-features = ["render"]` in `Cargo.toml`, so `cargo build --no-default-features` skips them and refuses to compile if requested explicitly.

### Sim hierarchy

`Simulation → Zones → Zone → { WorldTransform, Components }`, with a generic generational slot-map under everything:

- `SlotMap<K: SlotKey, V>` — generic generational storage.
- `WorldObjectId`, `ZoneId` — newtype keys; the type system rejects mismatched lookups.
- `WorldTransform` — per-object spatial state (position + rotation only). The struct is the *transform*; the *object* is the id plus whatever components are attached. Richer payloads live in `Components`, not on the transform.
- `Zone` — struct holding `objects: SlotMap<WorldObjectId, WorldTransform>` + `components: Components`. `Zone::remove` is the lifecycle choke point: it removes the object **and** cascades to `components.remove_all(id)`. `Zone::split_mut` returns independent borrows of the two when you need to iterate components and mutate objects in one pass.
- `Components` — heterogeneous, type-erased registry of sparse per-object state. `HashMap<TypeId, Box<dyn ComponentStorage>>` outer, `HashMap<WorldObjectId, T>` per type, lazy-allocated on first `insert::<T>`. APIs: `insert/get/get_mut/remove/iter/iter_mut`, all generic over `T: 'static`. Closer to RimWorld's `ThingComp` bag than to archetype ECS — the right shape for sim-game state where most facts are sparse and optional.
- `Zones = SlotMap<ZoneId, Zone>` (type alias). User's `Simulation` impl owns one.
- `WorldObjectRef { zone, id }` with `resolve(&zones)` / `resolve_mut` — fully-qualified cross-zone handle for camera targets, AI memory, save pointers.

### View hierarchy

The user implements the `View` trait with an associated `Sim: Simulation`:

- `const CONFIG: ViewConfig` — static window/render-target config: `title`, `clear_colour`, `depth_format`. Read by the engine *before* `init` runs, so pipelines built inside `init` can declare the same depth format the engine has allocated. New static knobs (MSAA samples, present mode, …) land here; new per-frame hooks land on `View`. Defaults to `ViewConfig::DEFAULT`; override with struct-update syntax (`..ViewConfig::DEFAULT`).
- `init(&Renderer) -> Self` — build pipelines, allocate buffers. The renderer is fully ready when this runs — including the depth attachment if `CONFIG.depth_format.is_some()`.
- `render(&self, &Sim, alpha, &Renderer, &mut RenderPass)` — read sim, record draw calls. `&Sim` is read-only by signature, structurally preventing sim mutation from the render path.
- `input(&mut self, &mut Sim, &mut EngineCtx, &WindowEvent)` — sim-mutating user actions go through here.
- `ui(&mut self, &mut Sim, &mut EngineCtx, &egui::Context)` — behind the `egui` feature; build the per-frame debug overlay. May mutate sim and engine context just like `input`.
- `active_zone(&self, &Sim) -> Option<ZoneId>` — which zone the camera is in. Default `None` is right for UI/2D views; world-space views typically return `self.camera.zone`. The engine uses this to drive `extract_environment` and (later) per-zone culling and streaming.
- `extract_environment(&self, &Sim, ZoneId) -> ViewEnvironment` — per-frame sim → GPU-friendly environment extraction. Engine calls it before `render`, writes the result into `Renderer::scene_bind_group`, and pipelines that declare `Renderer::scene_layout` read it automatically. Default returns `ViewEnvironment::neutral`. This is the same shape as visual extraction: sim owns facts (time of day), view owns appearance (sun direction + colour), engine drives the seam.
- `Camera` is a helper struct; the View opts in by holding one (UI/2D views don't need cameras). `Camera::zone: Option<ZoneId>` is the conventional place to stash the active zone so `active_zone` is a one-liner. The engine-standard `CameraUniformData` carries `view_proj` + `right`/`up` basis (for billboards) + `position` (so lit materials can compute view direction per fragment); the `CameraBinding` bgl is `VERTEX_FRAGMENT`-visible.

`run::<MyView>(sim)` wires it all up: creates the event loop, builds a `Renderer`, calls `init`, and dispatches events. `run_with_clock` takes a custom `SimClock`.

### Tick model

Fixed-tick (default 60 Hz) with an accumulator. The simulation always sees a constant `tick_period` regardless of speed; varying `SimClock::speed` only changes how many ticks fire per wall-clock second, which keeps sim logic deterministic at any playback rate. Pause is `set_speed(0.0)`. `MAX_TICKS_PER_FRAME = 16` prevents spiral-of-death. `SimClock::alpha()` returns `[0, 1]` interpolation factor for smooth motion (currently plumbed through but no example uses it).

### Render objects

Mostly landed: `RenderTemplate`, `RenderRegistry`, `SlotKind`/`SlotValue`/`SlotRouting`, `MeshPart`, `EmitterPart`, visual-bounds AABBs, hysteresis-culled `LiveRenderObjects`, and the engine-driven `RenderObjectPass` helper exist. Uniform-routed slot packing, nested templates, structural-override rules, and visual scripting are still on the design page below.

Drawable content is organised view-side into **render objects** — templates analogous to Unity prefabs or Godot sub-scenes, each owning a hierarchy of meshes, emitters, materials, and view-side resources. Templates are identified by `RenderId` and registered when the camera enters a zone. Sim objects carry a `RenderId` naming which template renders them; many sim objects share one template (every oak tree → `tree_oak`). Per-instance variation lives in transforms and **slots**. This is closer to UE's `PrimitiveSceneProxy` model than to per-frame extraction — sim hands the view an identity + state, the view holds the structure.

`(SimId, RenderId)` is the composite key for a live visual instance. Instances are created on first visibility and destroyed on cull or zone leave.

**Slots** are typed, named parameters declared by a template (think Godot's `@export`, Unreal's `UPROPERTY`). The schema is a closed `SlotKind` enum (`F32`, `Vec3`, `Color`, `Bool`, `AssetRef<T>`, …) — explicitly not a `Variant` / `Box<dyn Any>` bag. Each slot also declares a `SlotRouting` (`Instance` or `Uniform`) at template-build time so the engine picks the right packing strategy without runtime inference. Sim attaches per-object `SlotValues` as a sim component keyed by the parent `WorldObjectId`; the view reads them at render time. `with_slot(name, kind)` defaults to `SlotRouting::Instance`; `with_routed_slot(name, kind, routing)` is the explicit form.

**`Uniform` routing is a doc-only reservation for now.** The variant survives in the enum so adding the packing path later doesn't break the API, but `RenderTemplate::with_routed_slot` panics if asked for `SlotRouting::Uniform` — the failure surfaces at the template-builder site, not as a draw-time trap. Implement the packing path when the first consumer needs it; the current default of indexing a uniform array by `instance_index` in the shader is the v1 plan when that happens.

**Engine pass.** `RenderObjectPass` owns the per-frame walk: `declare_and_cull` walks zones, declares one live instance per sim object carrying a `RenderId` component, and culls against a frustum; `for_each_alive_part` / `for_each_alive` then iterate alive instances, validate each parent's `SlotValues` against the template schema, and invoke user closures for mesh parts (and emitter parts when needed). View code supplies a per-part callback that does the actual draw-attrib push — the engine owns the traversal, not the bucketing. Adding a slot or a part to a template requires touching only the template declaration + extract closure, never the per-frame plumbing.

**Nested templates** are allowed, with rules:
- Nested children are live references to other templates, not embedded snapshots — template edits propagate.
- Instance overrides are slot values only. **Structural overrides are forbidden** (no "this instance has one extra child"); make a new template instead. This is the source of most of Unity's prefab pain — avoid it by construction.
- Child slots are not auto-exposed up the tree — parents re-export deliberately, never automatically.

**Visual scripting** lives in Rust as `RenderBehavior`-style traits declared by templates. No scripting language, node graphs, or hot-reload — deliberately deferred until the engine ships. Visual scripts may only mutate view state.

**Material model** is three-tier:
- *Material template* — pipeline + bind-group layout + slot schema, registered once.
- *Material instance* — a bind group + uniform buffer bound to a template, cached or per-frame.
- *Per-instance attributes* — model matrix, tint, anything varying per drawn copy, packed into the instance buffer (the existing `mat4_instance_attributes` helper is the right shape).

Two materials exist today: `UnlitColoredMaterial` (position-only vertex, no lighting) and `PbrMaterial` (metallic-roughness, single directional light, albedo texture + scalar metallic/roughness; Cook-Torrance specular + Lambertian diffuse). There is **no `Material` trait yet** — the two share a structural pattern (template / instance / per-instance attribs) but not an interface. Add one when a third material kind makes the duplication painful, not before. Materials are not subclassed by what they draw (no `SpriteMaterial`/`MeshMaterial`); the contract with geometry is the instance-attribute layout.

**Vertex layouts are a closed set.** Same architectural move as `SlotKind`: a small enumerable list of canonical per-vertex structs (currently `PosNormalUv`; `TerrainVertex` is owned by the terrain mesher), declared as `Pod` structs with `attributes(start_location)` helpers. Materials statically demand one layout — no runtime attribute negotiation, no string keys. Adding a layout is a deliberate code change.

**Textures and samplers** live on the view side too:
- `Texture::from_rgba8` uploads RGBA8 bytes with a CPU-generated mip chain (box-filter downsampling — naive sRGB averaging; revisit when it bites). sRGB vs linear is a constructor flag.
- `SamplerKind` is a closed enum (`LinearRepeat`, `LinearClamp`, `NearestClamp`), holding a live `wgpu::Sampler` per variant in `SamplerRegistry`. Materials reference samplers by kind, not by raw handle.
- Loading from disk (image files, glTF) doesn't exist yet — `from_rgba8` is the only path; the example uses a procedural checkerboard.

**Environment** — directional lights, sky dome, weather, fog — lives outside the sim's object/zone world but is **sim-derived facts → view-derived appearance**:
- `SimEnvironment` (sim) owns the *facts*: `time_of_day`, `day`, `seconds_per_day`. Held by the user's `Simulation` impl, advanced in `tick`. Opt-in helper struct, same status as `Camera`. Provides `sun_direction_for(time_of_day) -> Vec3` as a trivial Z-up sun model.
- `ViewEnvironment` (view) owns the *appearance*: `sun_direction`, `sun_color`, `ambient`, `sky_color`. Engine-defined concrete struct because it backs a fixed bind-group layout. Future fog/weather lands here.
- `View::extract_environment(&sim, zone)` runs once per frame per active zone, producing a `ViewEnvironment`. The engine writes it into `Renderer::scene_bind_group` (a `SceneEnvironmentBinding` the `Renderer` owns); any pipeline that declared `Renderer::scene_layout` reads it automatically. This is engine-driven, not user-driven like `CameraBinding`, because every input the extract needs lives in the sim — the View just declares the mapping.
- Weather, sky domes, IBL probes, and multiple lights are all later additions to `ViewEnvironment`'s shape and the scene uniform.

**Pass-awareness is deferred.** Single forward pass for now; shadow/depth-prepass would introduce material × pass → pipeline (Unreal's Material Domain). Don't build the permutation matrix until a second pass actually exists.

## Architectural invariants

These are load-bearing — don't propose changes that violate them without checking first.

- **Sim is renderer-ignorant.** No `wgpu`/`winit` imports anywhere in the sim module tree (`src/sim.rs` + `src/sim/`). The build-level test for this is `cargo build --no-default-features`.
- **Storage is the source of truth for "where is this object."** `WorldTransform` does not carry a `ZoneId` field. Its zone is implicit in which `Zone` holds it. Same for objects within a zone — no denormalised location data.
- **Zones are coordinate-isolated.** Each zone has its own local frame; the engine provides no cross-zone positional math. Movement between zones is a storage operation (remove + insert), not a position update. (Considered an intermediate `Surface` layer for multi-floor buildings; rejected because isolated surfaces are the same shape as zones — multi-floor buildings become multi-zone with stair triggers.)
- **Single sim-wide tick.** No per-zone clocks. LOD-by-distance happens within the single tick by doing less work for distant zones, not by scheduling them differently.
- **Don't fuse sim and view.** No "Sprite component" on the sim object, no scene-graph parent/child on the sim side. Rendering data lives in the View, not on the sim object.
- **Component lifecycle is bound to object lifecycle.** `Zone::remove` is the only path that keeps the `Components` registry in sync with the object slot-map. `Zone::split_mut` exposes the inner `SlotMap` for split-borrow iteration, but removing through it bypasses `Components::remove_all` and leaks components — don't.
- **Component iteration is non-deterministic for now.** `Components` uses `HashMap` internally, which has a randomly-seeded hasher in std. Iteration order varies across runs. Acceptable while prototyping; will need to swap for a sparse-set or fixed-seed hasher before sim replay / lockstep networking is on the table.
- **View state is recoverable from sim state.** Render objects are ephemeral. Zone leave tears down all view state for that zone; revisiting reconstructs it from sim state. Visual scripts may carry per-instance state but it must be derivable (or acceptably reset-on-re-view) from sim state — no view-side history. Cost: long-lived transients (smoke columns, fire flicker) cold-start when revisited. Acceptable for now; pre-warming visual state from sim history is the escape hatch if it becomes visible.
- **Visual bounds differ from sim bounds.** Render-object templates declare their own AABB encompassing emitter reach and other large effects. Visibility culling uses that, not the sim object's footprint AABB. A 0.5 m campfire with a 6 m smoke column has a visual AABB that includes the column.
- **Culling has hysteresis.** Within-zone visibility culling keeps an instance alive for ~30 frames after it leaves the visual-AABB-vs-frustum test, to avoid pop-out at grazing camera angles. Zone-level cull (camera in zone) is the coarse gate; visual-AABB-with-hysteresis is the fine gate.

## wgpu 29 / winit 0.30 quirks

The codebase pins to wgpu 29.0.3 and winit 0.30.13. Recent API changes that have caught out the existing examples:

- `wgpu::Instance::new` takes `InstanceDescriptor` by value, not by reference.
- Use `wgpu::InstanceDescriptor::new_without_display_handle_from_env()` (or with-display variants) — there is no `default()`.
- `Surface::get_current_texture()` returns the `CurrentSurfaceTexture` enum (`Success | Suboptimal | Outdated | Lost | Timeout | Occluded | Validation`), not `Result<SurfaceTexture, SurfaceError>`. Match all variants.
- `RenderPassDescriptor` requires `multiview_mask: Option<NonZeroU32>` (use `..Default::default()` if you don't need it).
- `RenderPipelineDescriptor` field is `multiview_mask`, not `multiview`.
- `RenderPassColorAttachment` has a `depth_slice` field (use `None` for 2D).
- `PipelineLayoutDescriptor` no longer has `push_constant_ranges`; it has `immediate_size`. Use `..Default::default()`.
- `DeviceDescriptor` has `experimental_features` and `trace` fields. Use `..Default::default()` for non-essential setup.
- `PipelineLayoutDescriptor::bind_group_layouts` is `&[Option<&BindGroupLayout>]` — wrap entries in `Some(...)`.
- `DepthStencilState::depth_write_enabled` is `Option<bool>` (not `bool`) and `depth_compare` is `Option<CompareFunction>` (not `CompareFunction`). Wrap both in `Some(...)`.
- winit 0.30 uses `ApplicationHandler` trait pattern (`run_app(&mut handler)`), not the old closure-based `run`.

When adding render code, copy from `examples/camera.rs` (instance buffer + uniforms) or `examples/triangle.rs` (no buffers) rather than referring to older wgpu tutorials.

## Conventions

- Edition 2024.
- Examples are runnable demos that exercise specific subsystems; the sim/view boundary is preserved even in examples (sim types in the `Sim` field of the user's `Game` struct, view-side state in the `View` impl).
- Tests live in `#[cfg(test)] mod tests` blocks within the file under test, placed in the slice that owns the public API being asserted. Sim-side modules are well-covered (`zone`, `components`, `terrain`, `environment`); render-side tests cover what doesn't need a live GPU (`Pod` layout sizes, vertex strides, mip math, CPU downsampling) and the rest is exercised by running examples manually.
- Re-export third-party crates from `currawong` (`glam`, `wgpu`, `winit` under `render`) so consumers don't need to pin versions themselves.
