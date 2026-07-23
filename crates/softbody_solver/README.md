# softbody_solver

rust-gpu XPBD softbody compute (Jacobi / Gauss–Seidel). SPIR-V is produced by the repo root [`build.rs`](../../build.rs). The same `common` / `types` modules compile on the host for unit tests.

## Features

- `solver-jacobi` (default) — parallel Jacobi distance + collision + normals
- `solver-gauss-seidel` — colored GS batches (`gs_edges` + dynamic uniform)

## Tests

```bash
cd crates/softbody_solver && unset RUSTUP_TOOLCHAIN && cargo test
```

## Notes

- Pin `glam = "=0.30.8"` and use toolchain `nightly-2026-04-11` (see root `rust-toolchain.toml` / `mise.toml`).
- Entry points need `#[unsafe(no_mangle)]`; SPIR-V names are module-qualified (`jacobi::…` / `gs::…`).
- Storage buffers that the host bind layout marks read-write must be `&mut` in every entry that uses them (Naga access-mode matching).
