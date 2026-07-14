# bevy_softbody

[![Crates.io](https://img.shields.io/crates/v/bevy_softbody.svg)](https://crates.io/crates/bevy_softbody)
[![Docs.rs](https://docs.rs/bevy_softbody/badge.svg)](https://docs.rs/bevy_softbody)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU cloth simulation for [Bevy](https://bevyengine.org/), using WebGPU compute shaders. Implements [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf) (eXtended Position Based Dynamics; Macklin et al., 2016).

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

Shaders ship embedded with the crate — dependents do not need to copy WGSL into their `assets/` folder.

## Run the demo

```bash
cargo run --example cloth
```

The example simulates a hanging cloth sheet with mouse grab and an egui panel for solver parameters.

**Gauss–Seidel solver** (optional, instead of the default parallel Jacobi):

```bash
cargo run --example cloth --no-default-features --features solver-gauss-seidel
```

## Features

- XPBD distance constraints on the GPU (stretch, shear, bend)
- Extended PBR material that reads simulated positions from GPU buffers
- Colored Gauss–Seidel or parallel Jacobi constraint solvers
- CPU reference solver and GPU/CPU parity tests

## Docs

- [Cloth simulation stability](docs/CLOTH_SIM_STABILITY.md)
- [Metal GPU profiling (xctrace)](docs/XCTRACE_EXPORT.md)

## License

Licensed under the [MIT License](LICENSE).
