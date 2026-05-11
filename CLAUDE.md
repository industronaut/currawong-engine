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

- `src/sim.rs` + `src/sim/` (submodules `slot_map`, `zone`, `components`, `clock`) — sim layer. `sim.rs` is a thin parent that owns the `Simulation` trait and re-exports the submodules' public types so callers see a flat surface. Always compiled. Depends only on `glam` and `std`. Never imports `wgpu` or `winit`.
- `src/render.rs` — view layer. Compiled only with the `render` feature (default on). Owns all GPU + windowing.
- `src/lib.rs` — re-exporter. Conditionally exposes the render layer behind `#[cfg(feature = "render")]`.

The `render` Cargo feature gates `pollster`, `wgpu`, `winit`, `bytemuck`, and `glam/bytemuck`. Render-side examples declare `required-features = ["render"]` in `Cargo.toml`, so `cargo build --no-default-features` skips them and refuses to compile if requested explicitly.

### Sim hierarchy

`Simulation → Zones → Zone → { WorldObject, Components }`, with a generic generational slot-map under everything:

- `SlotMap<K: SlotKey, V>` — generic generational storage.
- `WorldObjectId`, `ZoneId` — newtype keys; the type system rejects mismatched lookups.
- `Zone` — struct holding `objects: SlotMap<WorldObjectId, WorldObject>` + `components: Components`. `Zone::remove` is the lifecycle choke point: it removes the object **and** cascades to `components.remove_all(id)`. `Zone::split_mut` returns independent borrows of the two when you need to iterate components and mutate objects in one pass.
- `Components` — heterogeneous, type-erased registry of sparse per-object state. `HashMap<TypeId, Box<dyn ComponentStorage>>` outer, `HashMap<WorldObjectId, T>` per type, lazy-allocated on first `insert::<T>`. APIs: `insert/get/get_mut/remove/iter/iter_mut`, all generic over `T: 'static`. Closer to RimWorld's `ThingComp` bag than to archetype ECS — the right shape for sim-game state where most facts are sparse and optional.
- `Zones = SlotMap<ZoneId, Zone>` (type alias). User's `Simulation` impl owns one.
- `WorldObjectRef { zone, id }` with `resolve(&zones)` / `resolve_mut` — fully-qualified cross-zone handle for camera targets, AI memory, save pointers.

### View hierarchy

The user implements the `View` trait with an associated `Sim: Simulation`:

- `init(&Renderer) -> Self` — build pipelines, allocate buffers.
- `render(&self, &Sim, alpha, &Renderer, &mut RenderPass)` — read sim, record draw calls. `&Sim` is read-only by signature, structurally preventing sim mutation from the render path.
- `input(&mut self, &mut Sim, &mut EngineCtx, &WindowEvent)` — sim-mutating user actions go through here.
- `depth_format() -> Option<TextureFormat>` — opt in to engine-managed depth. When `Some`, the renderer allocates a depth texture, recreates it on resize, and pre-attaches it to the frame's render pass. Pipelines must declare the same format in their `DepthStencilState`. Default `None` is right for 2D / UI views drawing in clip space.
- `Camera` is a helper struct; the View opts in by holding one (UI/2D views don't need cameras).

`run::<MyView>(sim)` wires it all up: creates the event loop, builds a `Renderer`, calls `init`, and dispatches events. `run_with_clock` takes a custom `SimClock`.

### Tick model

Fixed-tick (default 60 Hz) with an accumulator. The simulation always sees a constant `tick_period` regardless of speed; varying `SimClock::speed` only changes how many ticks fire per wall-clock second, which keeps sim logic deterministic at any playback rate. Pause is `set_speed(0.0)`. `MAX_TICKS_PER_FRAME = 16` prevents spiral-of-death. `SimClock::alpha()` returns `[0, 1]` interpolation factor for smooth motion (currently plumbed through but no example uses it).

### Render objects (planned)

Drawable content is organised view-side into **render objects** — templates analogous to Unity prefabs or Godot sub-scenes, each owning a hierarchy of meshes, emitters, materials, and view-side resources. Templates are identified by `RenderId` and registered when the camera enters a zone. Sim objects carry a `RenderId` naming which template renders them; many sim objects share one template (every oak tree → `tree_oak`). Per-instance variation lives in transforms and **slots**. This is closer to UE's `PrimitiveSceneProxy` model than to per-frame extraction — sim hands the view an identity + state, the view holds the structure.

`(SimId, RenderId)` is the composite key for a live visual instance. Instances are created on first visibility and destroyed on cull or zone leave.

**Slots** are typed, named parameters declared by a template (think Godot's `@export`, Unreal's `UPROPERTY`). The schema is a closed `SlotKind` enum (`F32`, `Vec3`, `Color`, `Bool`, `AssetRef<T>`, …) — explicitly not a `Variant` / `Box<dyn Any>` bag. Sim provides slot values per `(SimId, RenderId)`; the view routes them into uniforms or per-instance attribute buffers depending on cost.

**Nested templates** are allowed, with rules:
- Nested children are live references to other templates, not embedded snapshots — template edits propagate.
- Instance overrides are slot values only. **Structural overrides are forbidden** (no "this instance has one extra child"); make a new template instead. This is the source of most of Unity's prefab pain — avoid it by construction.
- Child slots are not auto-exposed up the tree — parents re-export deliberately, never automatically.

**Visual scripting** lives in Rust as `RenderBehavior`-style traits declared by templates. No scripting language, node graphs, or hot-reload — deliberately deferred until the engine ships. Visual scripts may only mutate view state.

**Material model** is three-tier:
- *Material template* — pipeline + bind-group layout + slot schema, registered once.
- *Material instance* — a bind group + uniform buffer bound to a template, cached or per-frame.
- *Per-instance attributes* — model matrix, tint, anything varying per drawn copy, packed into the instance buffer (the existing `mat4_instance_attributes` helper is the right shape).

Base `Material` is opaque — a typed bind-group producer. PBR (albedo/metallic/roughness) is `PbrMaterial: Material`, not baked into the base abstraction. Materials are not subclassed by what they draw (no `SpriteMaterial`/`MeshMaterial`); the contract with geometry is the instance-attribute layout.

**Environment** — directional lights, sky dome, weather, fog — lives outside the sim entirely. It's view-owned global state attached to the camera/scene, not driven by zone object lists or `RenderId`. Registered alongside a zone but not produced by extraction.

**Pass-awareness is deferred.** Single forward pass for now; shadow/depth-prepass would introduce material × pass → pipeline (Unreal's Material Domain). Don't build the permutation matrix until a second pass actually exists.

## Architectural invariants

These are load-bearing — don't propose changes that violate them without checking first.

- **Sim is renderer-ignorant.** No `wgpu`/`winit` imports anywhere in the sim module tree (`src/sim.rs` + `src/sim/`). The build-level test for this is `cargo build --no-default-features`.
- **Storage is the source of truth for "where is this object."** `WorldObject` does not carry a `ZoneId` field. Its zone is implicit in which `Zone` holds it. Same for objects within a zone — no denormalised location data.
- **Zones are coordinate-isolated.** Each zone has its own local frame; the engine provides no cross-zone positional math. Movement between zones is a storage operation (remove + insert), not a position update. (Considered an intermediate `Surface` layer for multi-floor buildings; rejected because isolated surfaces are the same shape as zones — multi-floor buildings become multi-zone with stair triggers.)
- **Single sim-wide tick.** No per-zone clocks. LOD-by-distance happens within the single tick by doing less work for distant zones, not by scheduling them differently.
- **Don't fuse sim and view.** No "Sprite component on WorldObject", no scene-graph parent/child on the sim side. Rendering data lives in the View, not the WorldObject.
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
- Tests live in `#[cfg(test)] mod tests` blocks within the file under test, placed in the slice that owns the public API being asserted (currently `src/sim/zone.rs` and `src/sim/components.rs`). Render-side code has no tests yet — it's covered by running examples manually.
- Re-export third-party crates from `currawong` (`glam`, `wgpu`, `winit` under `render`) so consumers don't need to pin versions themselves.
