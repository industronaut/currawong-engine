# examples/CLAUDE.md

Per-example state for quick discovery. The root [`CLAUDE.md`](../CLAUDE.md) owns
architectural invariants and is the right place for cross-cutting facts —
this file is the index of *what each example currently demonstrates*, which
changes often as features land.

If something here contradicts an example's own module docs, the module docs
are authoritative (they're edited by the PR that lands the change). Update
this file when an example crosses a feature boundary worth knowing about
without reading the file.

## The two reference examples

**[`lumber_camp/`](lumber_camp/)** — the end-to-end game PoC. Five sim
submodules (`motion`, `chopping`, `hauling`, `idle`) under
[`sim/`](lumber_camp/sim/), four view submodules (`pawn`, `tree`, `debugui`,
`gameui`) under [`render/`](lumber_camp/render/). This is the canonical
reference for:

- **Kind → template binding.** Sim attaches `KindId` components; view walks
  `Definitions` at init and registers one `RenderTemplate` per kind with a
  `render` block. Per-instance update dispatches on a precomputed
  `HashMap<KindId, RenderShape>`. See
  [`render/mod.rs`](lumber_camp/render/mod.rs) header comment for the full
  convention.
- **Sun-cascaded shadows.** `ShadowMeshPipeline` + `SunCascades` driving a
  3-cascade CSM. The cascade refresh pattern (cascade-0 every frame, 1/2
  staggered) is in `render/mod.rs`.
- **GPU picking with multi-kind dispatch.** Per-instance `with_hit_id` for
  pawns/trees, per-cell ids from the terrain renderer; one
  `Renderer::hit_id_hover` call demuxes to both.
- **yakui game UI.** [`render/gameui.rs`](lumber_camp/render/gameui.rs) is
  the only example using the shipped-game UI seam (NineSlice panels, font
  loading, click handlers pushing commands). Debug overlay (egui) lives
  alongside it in [`render/debugui.rs`](lumber_camp/render/debugui.rs).
- **Asset streaming through `AssetServer`** with magenta fallback. Hold `F`
  to force every handle to look `Loading`.

Sim mutation goes through `Command` (`ToggleDesignation`). The win/lose
state machine in `sim/mod.rs` freezes the world on `Won`/`Lost` — no
in-game restart.

**[`lumber_editor/`](lumber_editor/)** — single-item kind viewer with
debug overlays. Started life as a glb/texture viewer (#120); now also the
visualisation surface for kind metadata. Split across [`main.rs`](lumber_editor/main.rs)
(View struct + `View` trait impl + entry point), [`sim.rs`](lumber_editor/sim.rs)
(Command + Game), [`scene.rs`](lumber_editor/scene.rs) (checker floor +
facing arrow), [`overlays.rs`](lumber_editor/overlays.rs) (bounds +
interaction-tile + footprint-tile fat-line overlays),
[`kind_panel.rs`](lumber_editor/kind_panel.rs) (left-side egui surface),
[`hot_reload.rs`](lumber_editor/hot_reload.rs) (file-watcher pump +
template rebuild), [`mesh_edit.rs`](lumber_editor/mesh_edit.rs) (recalc /
Save / auto-frame trio). Current state:

- **Kind dropdown + auto-framing.** Left egui panel has an `egui::ComboBox`
  over every kind with a `render` block; selecting one swaps the displayed
  object and re-frames the orbit camera to that kind's
  `bounds_min`/`bounds_max` AABB.
- **Checkerboard ground + shadows** (#121). Same `ShadowMeshPipeline` /
  `SunCascades` as `lumber_camp`; the ground plane is a procedural
  `Texture` baked at init.
- **Yellow visual-bounds overlay** (#123). `FatLineMaterial` wireframe of
  the selected kind's visual AABB at 2.5 px screen-space.
- **Green interaction-tiles overlay** (#124). One fat-line square per tile
  in `Interaction::tiles(transform)`. Kinds whose def omits `interaction:`
  parse to `Interaction::None` and draw zero tiles.
- **Orange footprint-tiles overlay.** Same shape as the interaction
  overlay but with two diagonals across each square, one per tile in
  `Footprint::tiles(transform)`. Kinds whose def omits `footprint:` parse
  to an empty `Footprint` and draw zero tiles. The 2×2 placement
  footprint on `currawong:lumber_camp` is the canonical multi-tile demo;
  the trees declare a 1×1 footprint.
- **Visibility checkboxes** in the same left panel toggle the main item
  mesh (and its shadow), the bounding-box overlay, the interaction-tiles
  overlay, and the footprint-tiles overlay independently.
- **3D transform gizmo** ([`gizmo.rs`](lumber_editor/gizmo.rs)) on the
  selected scene node via `transform-gizmo-egui`. Drives the same
  `local_transform` as the inspector DragValues, so dirty-tracking and
  Save pick gizmo drags up for free. Left-panel radio group toggles
  Translate / Rotate / Scale / All. Walks the parent chain to compose
  world space, bakes the gizmo result back to local on edit.

Sim is one zone with one object at the origin; sole mutation is
`Command::SelectKind`. `Game::render_specs` and `Game::interactions` cache
parsed-at-startup typed values per kind so the per-frame path is one
HashMap lookup.

The editor doubles as the minimal reference for "build a view that reads
kinds and streams their assets" without the full lumber_camp scaffolding
(no terrain, no picking, no yakui).

## The other examples

Grouped by what they're the canonical reference for. All keep the
sim/view boundary even when minimal.

**Render layer primitives**
- [`materials.rs`](materials.rs) — `UnlitColoredMaterial` template +
  instance + per-instance attribs in one file. The minimal three-tier
  material reference.
- [`textured_pbr.rs`](textured_pbr.rs) — five PBR cubes varying
  metallic/roughness, sun driven by `SimEnvironment::time_of_day`. Run
  with `--features egui` for the time-of-day slider.
- [`campfire.rs`](campfire.rs) — mesh + particle emitters with
  `EmitterReconciler`. Demonstrates lit-state lifecycle: toggling fire
  off stops emission but lets in-flight particles linger.

**Asset pipeline**
- [`assets.rs`](assets.rs) — `AssetServer` streaming. Two cubes; one
  streams texture only, one streams mesh + texture. `F` forces the
  magenta fallback.
- [`blender_import.rs`](blender_import.rs) — multi-primitive glb with
  Blender-authored material slot names resolved through
  `MaterialRegistry`. On a fresh checkout the slot misses → magenta
  (deliberately loud).

**Terrain meshers**
- [`trees.rs`](trees.rs) — ~200 trees on a square grid; live sim mutation
  (age → height/tint) and GPU picking (per-instance hit ids + per-cell
  hit ids in one demux).
- [`hex_terrain.rs`](hex_terrain.rs) — same `FlatTopsMesher` over
  `HexGrid` instead of square grid. Topology swap, mesher unchanged.
- [`slope_terrain.rs`](slope_terrain.rs) — `SlopeMesher`; corners sit at
  `max(floor_height)` of touching cells. Transport Tycoon aesthetic.

**Sim/view boundary tests**
- [`headless.rs`](headless.rs) — sim ticks with no window. The
  build-system test is `cargo run --example headless
  --no-default-features` — that's the architectural assertion.
- [`multi_zone.rs`](multi_zone.rs) — two coordinate-isolated zones, stair
  tile triggers `Zone::remove` + `Zone::insert`. Different stair coords
  per zone underline the local-frame rule. Per-zone
  `extract_environment` makes the active-zone swap visible.

## Conventions specific to examples

- Each example mounts its own `Vfs` per side. Sim mounts one to load
  `Definitions`; view mounts another for `AssetServer`. Same on-disk
  content, independent caches — there's no engine-level requirement that
  they share an instance. Mods would mount layers on top.
- Sim-side files use `mod.rs` + submodules per behaviour
  (`chopping.rs`, `hauling.rs`, …). View-side files split per kind
  (`pawn.rs`, `tree.rs`) and per UI surface (`debugui.rs`, `gameui.rs`).
  This is the vertical-slice pattern the root CLAUDE.md alludes to.
- Per-kind material/mesh setup that's shared between `lumber_camp` and
  `lumber_editor` lives in `src/render/` engine helpers, not in the
  examples:
  - `PbrMaterial::streamed_kind_body_templates` — walks `Definitions`,
    builds one streamed `MeshTemplate` per kind with a `render` block.
  - `MeshDraw::pbr_with_atlas` / `MeshDraw::depth_only` — per-primitive
    pipeline-switching draw loops shared between the colour pass and the
    shadow pass.
  - `MeshDraw::refresh_pbr_atlas_materials` — the cascade-0 phase-1.5
    refresh + adjustment-cache loop.
  If you find yourself copy-pasting between the two examples, that's the
  signal to land another helper in `src/render/`.
- `required-features` in `Cargo.toml` gates render/egui/yakui examples
  behind their features so `cargo build --no-default-features` doesn't
  try to compile them.
