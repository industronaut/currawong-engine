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

cargo run --example lumber_camp          # end-to-end game loop: pawns chop trees, haul logs to a stockpile
cargo run --example trees                # ~200 trees growing under live sim mutation
cargo run --example textured_pbr         # PBR cubes lit by a sim-driven sun
cargo run --example textured_pbr --features egui   # same, with debug overlay
cargo run --example campfire             # mesh + particle emitters with lit-state lifecycle
cargo run --example materials            # material template / instance / per-instance attrib pattern
cargo run --example assets               # streaming asset pipeline; async mesh + texture loads through AssetServer
cargo run --example blender_import       # glTF 2.0 mesh + material import authored in Blender
cargo run --example multi_zone           # two zones + stair trigger; coordinate-isolated rendering
cargo run --example hex_terrain          # hex topology through the same flat-tops mesher
cargo run --example slope_terrain        # sloped mesher with height-aware GPU picking
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

The central commitment is **sim/view separation**, modelled on UE-style proxy extraction rather than Unity/Godot scene-graph integration. The codebase splits into three top-level modules:

- `src/sim.rs` + `src/sim/` (submodules `slot_map`, `grid`, `zone`, `components`, `clock`, `command`, `terrain`, `environment`) — sim layer. `sim.rs` is a thin parent that owns the `Simulation` trait and re-exports the submodules' public types so callers see a flat surface. Always compiled. Depends only on `glam` and `std`. Never imports `wgpu` or `winit`.
- `src/data.rs` + `src/data/` (submodules `path`, `source`, `fs_source`, `memory_source`, `vfs`, `definitions`) — data layer. The virtual filesystem (`Vfs` over a stack of `AssetSource` layers) and the RON-backed `Definitions` registry of namespaced kinds. Sim consumes `Definitions`; the view streams assets through the same VFS. Always compiled, no `wgpu`/`winit` dependencies; the I/O surface is shaped so a WASM port swaps the bottom layer rather than rewriting.
- `src/render.rs` + `src/render/` — view layer. Compiled only with the `render` feature (default on). Owns all GPU + windowing.
- `src/lib.rs` — re-exporter. Conditionally exposes the render layer behind `#[cfg(feature = "render")]`.

The build-system-enforced sim/view boundary is the `render` Cargo feature, which gates `pollster`, `wgpu`, `winit`, `bytemuck`, `gltf`, `image`, and `glam/bytemuck`. Render-side examples declare `required-features = ["render"]` (or `["yakui"]`) in `Cargo.toml`, so `cargo build --no-default-features` skips them and refuses to compile if requested explicitly.

### Sim hierarchy

`Simulation → Zones → Zone → { WorldTransform, Components }`, with a generic generational slot-map under everything:

- `SlotMap<K: SlotKey, V>` — generic generational storage.
- `WorldObjectId`, `ZoneId` — newtype keys; the type system rejects mismatched lookups.
- `WorldTransform` — per-object spatial state (position + rotation only). The struct is the *transform*; the *object* is the id plus whatever components are attached. Richer payloads live in `Components`, not on the transform.
- `Zone` — struct holding `objects: SlotMap<WorldObjectId, WorldTransform>` + `components: Components`. `Zone::remove` is the lifecycle choke point: it removes the object **and** cascades to `components.remove_all(id)`. `Zone::split_mut` hands out a `WorldObjectsMut` wrapper + `&mut Components` when you need to iterate components and mutate objects in one pass; the wrapper omits `remove` so the choke point can't be bypassed.
- `Components` — heterogeneous, type-erased registry of sparse per-object state. `HashMap<TypeId, Box<dyn ComponentStorage>>` outer, `HashMap<WorldObjectId, T>` per type, lazy-allocated on first `insert::<T>`. APIs: `insert/get/get_mut/remove/iter/iter_mut`, all generic over `T: 'static`. Closer to RimWorld's `ThingComp` bag than to archetype ECS — the right shape for sim-game state where most facts are sparse and optional.
- `Zones = SlotMap<ZoneId, Zone>` (type alias). User's `Simulation` impl owns one.
- `WorldObjectRef { zone, id }` with `resolve(&zones)` / `resolve_mut` — fully-qualified cross-zone handle for camera targets, AI memory, save pointers.

### View hierarchy

The user implements the `View` trait with an associated `Sim: Simulation`:

- `const CONFIG: ViewConfig` — static window/render-target config: `title`, `clear_colour`, `depth_format`. Read by the engine *before* `init` runs, so pipelines built inside `init` can declare the same depth format the engine has allocated. New static knobs (MSAA samples, present mode, …) land here; new per-frame hooks land on `View`. Defaults to `ViewConfig::DEFAULT`; override with struct-update syntax (`..ViewConfig::DEFAULT`).
- `init(&Renderer) -> Self` — build pipelines, allocate buffers. The renderer is fully ready when this runs — including the depth attachment if `CONFIG.depth_format.is_some()`.
- `render(&self, &Sim, alpha, &Renderer, &mut RenderPass)` — read sim, record draw calls. `&Sim` is read-only by signature, structurally preventing sim mutation from the render path.
- `input(&mut self, &Sim, &mut EngineCtx, &mut CommandQueue<Sim::Command>, &WindowEvent)` — user input events. Sim mutation is reified through `Command` values pushed into `cmds`; `&Sim` is read-only so the only path to mutating sim from a view is the queue.
- `update(&mut self, &Sim, &mut EngineCtx, &mut CommandQueue<Sim::Command>, dt: Duration)` — per-frame view-side update. The engine calls it once per frame, *after* sim ticking and *before* `extract_environment` / `render`. `dt` is **wall-clock**, not sim time — view animation keeps running at sim speed 0 or 3×. `&Sim` is read-only by signature, mirroring `render`. Continuous interactions that need to drive sim state (held-key intents, drag-paint strokes) coalesce into one `Command` here at stroke end and push into `cmds`. Default no-op.
- `ui(&mut self, &Sim, &mut EngineCtx, &mut CommandQueue<Sim::Command>, &egui::Context)` — behind the `egui` feature; build the per-frame debug overlay. Widgets that need to mutate sim state push commands instead of taking `&mut Sim`.
- `game_ui(&mut self, &Sim, &mut EngineCtx, &mut CommandQueue<Sim::Command>)` — behind the `yakui` feature; build the per-frame shipped game UI (HUDs, menus, panels). Widget calls (`yakui::label`, `yakui::button`, …) attach to the engine's `Yakui` state via yakui's thread-local context. Independent of `ui` — both features can be enabled together. Defaults to no-op.
- `active_zone(&self, &Sim) -> Option<ZoneId>` — which zone the camera is in. Default `None` is right for UI/2D views; world-space views typically return `self.camera.zone`. The engine uses this to drive `extract_environment` and (later) per-zone culling and streaming.
- `extract_environment(&self, &Sim, ZoneId) -> ViewEnvironment` — per-frame sim → GPU-friendly environment extraction. Engine calls it before `render`, writes the result into `Renderer::scene_bind_group`, and pipelines that declare `Renderer::scene_layout` read it automatically. Default returns `ViewEnvironment::neutral`. This is the same shape as visual extraction: sim owns facts (time of day), view owns appearance (sun direction + colour), engine drives the seam.
- `Camera` is a helper struct; the View opts in by holding one (UI/2D views don't need cameras). `Camera::zone: Option<ZoneId>` is the conventional place to stash the active zone so `active_zone` is a one-liner. The engine-standard `CameraUniformData` carries `view_proj` + `right`/`up` basis (for billboards) + `position` (so lit materials can compute view direction per fragment); the `CameraBinding` bgl is `VERTEX_FRAGMENT`-visible.
- **Camera rigs** are input-driven controllers that *drive* a `Camera`, separate from the camera helper itself. Today there's one: `OrbitRig` — strategy-game-style orbit around a focal point with RMB-drag rotation, WASD pan, and scroll zoom. The View holds both a `Camera` and an `OrbitRig`; route events to `rig.handle_event` in `input`, call `rig.update(dt)` + `rig.apply_to(&mut camera)` in `update`. Designed as a value type the View can hold one or more of — Cinemachine-style blending/cuts between rigs is the planned forward direction, not bundling everything into one fatter camera.

`run::<MyView>(sim)` wires it all up: creates the event loop, builds a `Renderer`, calls `init`, and dispatches events. `run_with_clock` takes a custom `SimClock`.

### Tick model

Fixed-tick (default 60 Hz) with an accumulator. The simulation always sees a constant `tick_period` regardless of speed; varying `SimClock::speed` only changes how many ticks fire per wall-clock second, which keeps sim logic deterministic at any playback rate. Pause is `set_speed(0.0)`. `MAX_TICKS_PER_FRAME = 16` prevents spiral-of-death. `SimClock::alpha()` returns `[0, 1]` interpolation factor for smooth motion (currently plumbed through but no example uses it). `SimClock::tick()` returns the current sim tick as `u64` — part of sim state, serialised with saves, the same number `CommandQueue` keys `apply_at_tick` against.

### Sim mutation: `Command` + `CommandQueue`

External mutation of the sim — input handlers, UI widgets, scripted tests, eventual network broadcasts — goes through one seam: the user defines `Simulation::Command` (a closed enum of intents), View callbacks push `Command` values into a `CommandQueue<Sim::Command>`, and the engine drains the queue at each sim tick boundary calling `Sim::apply_command(&cmd)` once per ready command before `Sim::tick`. No code outside the engine ever holds `&mut Sim`. This is the precondition for replay, save-as-command-log, scripted tests, undo, console commands, and (long-horizon) lockstep multiplayer.

- Each queued command carries `apply_at_tick: u64`. At sim tick N the engine drains every command with `apply_at_tick <= N` in insertion order; later-scheduled commands stay queued. Single-player code calls `cmds.push_now(cmd)` which stamps the queue's current tick (engine sets it at each tick boundary); future-scheduled `push(apply_at_tick, cmd)` exists from day one so the multiplayer migration doesn't have to touch every call site.
- **Commands are primitive data.** IDs (`WorldObjectId`, `KindId`, `ZoneId`), numbers, enums, owned strings. No `Arc<dyn Trait>`, no closures, no borrows. If a command needs to reach a sim object it carries the id and `apply_command` resolves it against current sim state at apply time. This is what makes the future serialise-for-wire / serialise-for-replay path trivial.
- **Granularity for continuous interactions.** A drag-to-paint-terrain shouldn't emit one Command per pixel per frame — coalesce to a single `PaintStroke { from, to, brush }` at stroke end. The line between view-only interaction (camera pan with held WASD lives in `update`, not sim) and sim mutation that needs coalescing is per-case; some examples won't fit cleanly and want a deliberate design rather than a general rule.
- Today's variants in the examples cover: pawn ToggleDesignation (lumber_camp), StepPlayer / ResetPlayer (multi_zone), ToggleFire (campfire), SetTimeOfDay (textured_pbr's egui slider). Sims with no external mutation (view-only demos, headless fixtures) default `type Command = ();`.
- **Out of scope for the Command landing, deferred to follow-up issues:** determinism enforcement (ordered `Components` storage, seeded PRNG, fixed-point math), replay recording/playback, networking transport, routing internal sim mutation through Commands (Factorio-style unified log).

### Render objects

Landed: `RenderTemplate`, `RenderRegistry`, `MeshPart`, `EmitterPart`, visual-bounds AABBs, hysteresis-culled `LiveRenderObjects`, the engine-driven `RenderObjectPass` helper, and the per-instance update hook + persistent view-side `RenderInstance` state (`LiveRenderObject` with `RenderPartState` per mesh/emitter part, holding `visible: bool` plus room for cached attribs and animation phase). Nested templates and visual scripting are deferred — see **Future directions** below.

Drawable content is organised view-side into **render objects** — templates analogous to Unity prefabs or Godot sub-scenes, each owning a hierarchy of meshes, emitters, materials, and view-side resources. Templates are identified by `RenderId` and registered when the camera enters a zone. Sim objects carry a `RenderId` naming which template renders them; many sim objects share one template (every oak tree → `tree_oak`). Per-instance variation lives in transforms and in the persistent view-side `LiveRenderObject` state that the update hook writes. This is closer to UE's `PrimitiveSceneProxy` model than to per-frame extraction — sim hands the view an identity + state, the view holds the structure.

`(SimId, RenderId)` is the composite key for a live visual instance. Instances are created on first visibility and destroyed on cull or zone leave.

**Sim→view publishing contract.** Sim state lives in typed [`Components`] keyed by `WorldObjectId` (`Tree { age_ticks, … }`, `ActiveAction::Hauling`, `Health`). The view reads these directly in the per-instance update hook below; there is no separate primitive-bag layer. Component names address sim facts in **sim vocabulary** — names like `IsVisible`, `TintColor`, or `MeshPart3Enabled` would be view-vocabulary leaking into the sim and are a category error: visibility and tint are *derived* from sim facts during the update, not *named* on the sim side.

**View per-instance state and update.** Each live `(SimId, RenderId)` pair has a persistent **render instance** on the view side (`LiveRenderObject`), holding per-part state (`RenderPartState { visible: bool }` for each mesh and emitter part; future cached attribs, animation phase, etc.). The template's update hook runs once per frame per instance, *before* extract, with the signature `(parent, render_id, &Components, &mut LiveRenderObject)`. This is the single seam where sim→view translation lives — e.g. `instance.mesh_parts[CRATE].visible = components.get::<ActiveAction>(id) == Some(&ActiveAction::Hauling)`. Visibility is **view-side state**; the sim only publishes the semantic fact and the template decides which parts that lights up. Pull every frame: dirty tracking has been considered and deliberately deferred until a profile justifies the bookkeeping cost.

**Engine pass.** `RenderObjectPass` owns the per-frame walk: `declare_and_cull` walks zones, declares one `LiveRenderObject` per sim object carrying a `RenderId` component, and culls against a frustum; the engine then invokes the per-instance update hook with the parent's `Components` + the persistent `LiveRenderObject`; finally `for_each_alive_part` / `for_each_alive` iterate alive instances and invoke user extract closures that read instance state only (no sim access during extract). View code supplies per-part extract callbacks that do the actual draw-attrib push — the engine owns the traversal, not the bucketing. Parts are gated by `instance.mesh_parts[i].visible` / `instance.emitter_parts[i].visible` (and the whole instance by `root_visible`). Adding a per-instance fact to a template requires touching only the update + extract closures, never the per-frame plumbing.

**Material model** is three-tier:
- *Material template* — pipeline + bind-group layout + slot schema, registered once.
- *Material instance* — a bind group + uniform buffer bound to a template, cached or per-frame.
- *Per-instance attributes* — model matrix, tint, anything varying per drawn copy, packed into the instance buffer (the existing `mat4_instance_attributes` helper is the right shape).

Four materials exist today, with a thin shared `MeshMaterial` trait (associated `Instance` type + `pipeline()` accessor; one accessor only, generic draw helpers land when a call site needs them):

- `UnlitColoredMaterial` — position-only vertex, no lighting; the minimal template/instance/per-instance-attrib reference.
- `PbrMaterial` — metallic-roughness, single directional light, albedo texture + scalar metallic/roughness; Cook-Torrance specular + Lambertian diffuse.
- `PbrAtlasMaterial` — stylized PBR variant that reads albedo + metallic/roughness/emissive from two shared atlas textures and resolves a per-instance atlas tile from a glb material slot; paired with `MaterialRegistry` (name-keyed `MaterialId` lookup, `namespace:name` grammar matching `KindId`, magenta-style fallback on miss).
- `TerrainMaterial` — opaque + transparent terrain pipelines over the canonical `TerrainVertex`. Not a `MeshMaterial` impl because it doesn't take the standard mesh-instance attribute layout.

Materials are not subclassed by what they draw beyond that mesh/terrain split — the contract with mesh geometry is the `MeshInstanceAttribs` layout, declared once and shared.

`MaterialInstanceRegistry<I, K>` (in `material.rs`) is the older enum-keyed shape — fine when the call site enumerates its materials at compile time. `MaterialRegistry` (in `material_registry.rs`) is the newer string-keyed shape needed once glb files started naming material slots as `currawong:mat_bark` — same role as `KindId` for sim kinds. Both live; pick by whether the lookup key is closed (enum) or open (glb-authored string).

**Vertex layouts are a closed set.** A small enumerable list of canonical per-vertex structs (currently `PosNormalUv`; `TerrainVertex` is owned by the terrain mesher), declared as `Pod` structs with `attributes(start_location)` helpers. Materials statically demand one layout — no runtime attribute negotiation, no string keys. Adding a layout is a deliberate code change.

**Textures and samplers** live on the view side too:
- `Texture::from_rgba8` uploads RGBA8 bytes with a CPU-generated mip chain (box-filter downsampling — naive sRGB averaging; revisit when it bites). sRGB vs linear is a constructor flag.
- `Texture::from_png_bytes_with_device` decodes PNG/JPEG (via the `image` crate) and uploads the same way. Loaded through the `AssetServer`.
- `SamplerKind` is a closed enum (`LinearRepeat`, `LinearClamp`, `NearestClamp`), holding a live `wgpu::Sampler` per variant in `SamplerRegistry`. Materials reference samplers by kind, not by raw handle.

**Streaming and assets.** Loading from disk goes through three cooperating pieces:
- `Vfs` (in `src/data/`) is the ordered stack of `AssetSource` layers — `FsSource` for native disk, `MemorySource` for tests / the future WASM `include_bytes!` archive. `VfsPath` is a normalised forward-slash newtype that rejects `..`, drive letters, backslashes, absolute paths; the grammar is enforced at the type level so mod-loaded layers can't escape the sandbox.
- `Mesh::from_gltf_bytes_with_device` / `decode_gltf_mesh` decode glTF 2.0 via the `gltf` crate, skipping `import`/`base64`/`image` features because bytes arrive through the VFS. Multi-primitive glb is supported; each primitive carries a material slot string that resolves through `MaterialRegistry`.
- `AssetServer` is the view-side gateway: hands out `Handle<T>` for typed asset paths (per-type, per-colour-space cache), spawns a `std::thread::spawn` background load per request that decodes and uploads directly to the GPU (wgpu 29 `Device`/`Queue` are `Send + Sync + Clone`), and serves a magenta-flavoured fallback (4×4 magenta texture + unit-cube placeholder mesh) for `Loading`/`Failed` handles. Carries a debug toggle (`set_force_loading`) that pins every handle to `Loading` so the fallback path is exercised in normal dev instead of rotting silently.

`Handle<T>` compares by identity, not contents — two loads of the same PNG are distinct assets from the caller's perspective. The slot is `OnceLock`-backed so the read path is lock-free after the loader writes.

**Environment** — directional lights, sky dome, weather, fog — lives outside the sim's object/zone world but is **sim-derived facts → view-derived appearance**:
- `SimEnvironment` (sim) owns the *facts*: `time_of_day`, `day`, `seconds_per_day`. Held by the user's `Simulation` impl, advanced in `tick`. Opt-in helper struct, same status as `Camera`. Provides `sun_direction_for(time_of_day) -> Vec3` as a trivial Z-up sun model.
- `ViewEnvironment` (view) owns the *appearance*: `sun_direction`, `sun_color`, `ambient`, `sky_color`. Engine-defined concrete struct because it backs a fixed bind-group layout. Future fog/weather lands here.
- `View::extract_environment(&sim, zone)` runs once per frame per active zone, producing a `ViewEnvironment`. The engine writes it into `Renderer::scene_bind_group` (a `SceneEnvironmentBinding` the `Renderer` owns); any pipeline that declared `Renderer::scene_layout` reads it automatically. This is engine-driven, not user-driven like `CameraBinding`, because every input the extract needs lives in the sim — the View just declares the mapping.
- Weather, sky domes, IBL probes, and multiple lights are all later additions to `ViewEnvironment`'s shape and the scene uniform.

### Future directions

Designed-but-not-implemented pieces. Each is the planned shape for when a real consumer needs it — not in code today, so don't treat them as available APIs.

- **Nested templates.** Templates will be allowed to reference other templates as children, with rules: nested children are live references (template edits propagate), instance overrides are *per-instance state only* (structural overrides are forbidden — make a new template instead), and child state is never auto-exposed up the tree (parents re-export deliberately). The structural-override rule is the source of most of Unity's prefab pain; avoiding it by construction is the whole point.
- **Visual scripting.** Per-template `RenderBehavior`-style traits, in Rust — no scripting language, node graphs, or hot-reload. Visual scripts may only mutate view state.
- **Pass-awareness.** Single forward pass for now; shadow / depth-prepass / forward-plus would introduce material × pass → pipeline (Unreal's Material Domain). Don't build the permutation matrix until a second pass actually exists.

## Architectural invariants

These are load-bearing — don't propose changes that violate them without checking first.

- **Sim is renderer-ignorant.** No `wgpu`/`winit` imports anywhere in the sim module tree (`src/sim.rs` + `src/sim/`). The build-level test for this is `cargo build --no-default-features`.
- **Storage is the source of truth for "where is this object."** `WorldTransform` does not carry a `ZoneId` field. Its zone is implicit in which `Zone` holds it. Same for objects within a zone — no denormalised location data.
- **Zones are coordinate-isolated.** Each zone has its own local frame; the engine provides no cross-zone positional math. Movement between zones is a storage operation (remove + insert), not a position update. (Considered an intermediate `Surface` layer for multi-floor buildings; rejected because isolated surfaces are the same shape as zones — multi-floor buildings become multi-zone with stair triggers.)
- **Single sim-wide tick.** No per-zone clocks. LOD-by-distance happens within the single tick by doing less work for distant zones, not by scheduling them differently.
- **Don't fuse sim and view.** No "Sprite component" on the sim object, no scene-graph parent/child on the sim side. Rendering data lives in the View, not on the sim object.
- **Sim mutation flows through `Command` + `CommandQueue` + `apply_command`.** View callbacks (`input`, `update`, `ui`, `game_ui`) receive `&Sim` (read-only) and `&mut CommandQueue<Sim::Command>`; the engine is the only thing that holds `&mut Sim`, applying commands at tick boundaries before `tick`. The exception is *engine-managed* state on `EngineCtx` — `clock.set_speed`, `event_loop.exit` — which is view/engine concerns, not sim mutation. Commands must be primitive serialisable data; if a variant carries an `Arc<dyn>` or a closure the design has slipped.
- **Sim `Component` names address sim facts, never view decisions.** A `Component` called `IsVisible`, `TintColor`, or `MeshPart3Enabled` is a category error — the view derives visibility and tint from sim facts during the per-instance update, but the sim never *names* view state. Components carry whatever shape the fact needs (custom enums like `ActiveAction = Moving | Hauling | Idle`, structs, primitives); the view reads them typed in the update hook.
- **All sim→view translation happens in the per-instance update.** Extract closures read only the persistent `RenderInstance`, never the sim directly. This keeps extract a pure GPU-attrib write step and contains the component → view-state mapping to one well-known place per frame. View-only state (hover tint, sway phase, animation blends) lives on `RenderInstance` too, alongside sim-derived state.
- **Component lifecycle is bound to object lifecycle.** `Zone::remove` is the only path that keeps the `Components` registry in sync with the object slot-map. `Zone::split_mut` returns a `WorldObjectsMut` wrapper that exposes lookup and mutation but not removal, so this invariant is enforced at the type level rather than by convention.
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

When adding render code, copy from `examples/lumber_camp/` (full pipeline + instance buffers + camera/scene bindings + picking) or `examples/materials.rs` (minimal unlit pipeline + per-instance attribs) rather than referring to older wgpu tutorials.

## Conventions

- Edition 2024.
- Examples are runnable demos that exercise specific subsystems; the sim/view boundary is preserved even in examples (sim types in the `Sim` field of the user's `Game` struct, view-side state in the `View` impl).
- Tests live in `#[cfg(test)] mod tests` blocks within the file under test, placed in the slice that owns the public API being asserted. Sim-side modules are well-covered (`zone`, `components`, `terrain`, `environment`); render-side tests cover what doesn't need a live GPU (`Pod` layout sizes, vertex strides, mip math, CPU downsampling) and the rest is exercised by running examples manually.
- Re-export third-party crates from `currawong` (`glam`, `wgpu`, `winit` under `render`) so consumers don't need to pin versions themselves.
