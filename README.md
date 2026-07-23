# bevy_softbody

[![Crates.io](https://img.shields.io/crates/v/bevy_softbody.svg)](https://crates.io/crates/bevy_softbody)
[![Docs.rs](https://docs.rs/bevy_softbody/badge.svg)](https://docs.rs/bevy_softbody)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU cloth simulation for [Bevy](https://bevyengine.org/). Compute solvers are [rust-gpu](https://rust-gpu.github.io/) SPIR-V; the cloth material stays WGSL `ExtendedMaterial`. Implements [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf) (eXtended Position Based Dynamics; Macklin et al., 2016).

![Cloth demo](assets/demo.jpg)

## Usage

Add the crate and register the compute + material plugins:

```toml
[dependencies]
bevy_softbody = "0.1"
```

```rust
use bevy::prelude::*;
use bevy_softbody::{
    cloth_compute::ClothComputePlugin,
    cloth_material::ClothMaterialPlugin,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((ClothComputePlugin, ClothMaterialPlugin))
        .run();
}
```

Compute SPIR-V is compiled at build time from [`crates/softbody_solver`](crates/softbody_solver); the vertex/material shader (`cloth_vertex.wgsl`) remains embedded WGSL.

## Build requirements

Building this crate (and the cloth example) needs the rust-gpu toolchain pin (`nightly-2026-04-11` + `rust-src` / `rustc-dev` / `llvm-tools`), declared in both [`mise.toml`](mise.toml) and [`rust-toolchain.toml`](rust-toolchain.toml).

With [mise](https://mise.jdx.dev/):

```bash
mise trust
mise install
mise exec -- cargo build
```

Or with rustup directly:

```bash
rustup toolchain install nightly-2026-04-11
rustup component add rust-src rustc-dev llvm-tools --toolchain nightly-2026-04-11
# If your shell exports another nightly via RUSTUP_TOOLCHAIN, clear it:
unset RUSTUP_TOOLCHAIN
cargo build
```

A different active nightly will fail `rustc_codegen_spirv`’s toolchain check. Native GPU only (Metal via Naga, or Vulkan); WebGPU/browser is not a goal of the SPIR-V path.

Solver math is tested on the host via the same `softbody_solver` helpers the GPU kernels use:

```bash
cd crates/softbody_solver && mise exec -- cargo test
# plus CPU cloth integration tests:
mise exec -- cargo test --lib
```

## Run the demo

```bash
mise exec -- cargo run --example cloth
# or: unset RUSTUP_TOOLCHAIN && cargo run --example cloth
```

The example simulates a hanging cloth sheet with mouse grab and an egui panel for solver parameters.

**Gauss–Seidel solver** (optional, instead of the default parallel Jacobi):

```bash
cargo run --example cloth --no-default-features --features solver-gauss-seidel
```

## Features

- XPBD distance constraints on the GPU (stretch, shear, bend) via rust-gpu SPIR-V
- Extended PBR material (WGSL) that reads simulated positions from GPU buffers
- Colored Gauss–Seidel or parallel Jacobi constraint solvers
- CPU reference solver (`xpbd_cpu`) and host unit tests of shared `softbody_solver` math

## Docs

- [Cloth simulation stability](docs/CLOTH_SIM_STABILITY.md)
- [Metal GPU profiling (xctrace)](docs/XCTRACE_EXPORT.md)


## License

Licensed under the [MIT License](LICENSE).
