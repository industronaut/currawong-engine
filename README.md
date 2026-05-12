# currawong

A Rust game engine being built from scratch, aimed at simulation-style games in the lineage of Dwarf Fortress, Factorio, and RimWorld. Named after the [Australian bird](https://en.wikipedia.org/wiki/Currawong).

**Status:** early. The engine itself is the deliverable — this is an architectural learning project, not a means of shipping a specific game. APIs will move.

## Design

The central commitment is **sim/view separation**, modelled on Unreal's proxy-extraction pattern rather than Unity/Godot scene-graph integration:

- **Sim layer** (`src/sim/`) — always compiled, depends only on `glam` and `std`. Owns world state: zones, world objects, sparse components, a fixed-tick clock. Knows nothing about rendering.
- **View layer** (`src/render.rs`) — compiled behind the `render` Cargo feature (default on). Owns `wgpu` + `winit`. Reads sim state each frame; never mutates it from the render path.

The boundary is enforced at build time: `cargo build --no-default-features` produces a sim-only binary with no GPU or windowing dependencies in the tree. That build succeeding is the architectural invariant test.

Read [CLAUDE.md](./CLAUDE.md) for the full architectural notes — sim hierarchy (`Simulation → Zones → Zone → { WorldTransform, Components }`), view hierarchy, the tick model, render-object templates and slots, the material model, and the load-bearing invariants the codebase commits to.

## Build

```bash
cargo build                              # full build (sim + render)
cargo build --no-default-features        # sim-only, no wgpu/winit
cargo test                               # all tests
cargo test --no-default-features         # sim tests only
cargo clippy --all-targets
```

## Examples

```bash
cargo run --example clear                # window with cleared background
cargo run --example triangle             # static colored triangle
cargo run --example input                # input demo + sim speed controls
cargo run --example camera               # sim/view extract + camera demo
cargo run --example instances            # instanced draws
cargo run --example campfire             # emitter reconciliation
cargo run --example materials            # material instances
cargo run --example render_objects       # render-object templates
cargo run --example headless             # sim ticking without any window
cargo run --example headless --no-default-features   # proves headless excludes wgpu/winit at compile time
```

## Setup

Activate the in-repo git hooks once per clone:

```bash
git config core.hooksPath .githooks
```

This wires up `.githooks/pre-commit`, which runs `cargo fmt --check` and blocks unformatted commits.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](./LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
