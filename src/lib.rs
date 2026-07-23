//! GPU XPBD cloth simulation with Bevy 0.19 (compute via rust-gpu SPIR-V).

#[cfg(all(not(feature = "solver-gauss-seidel"), not(feature = "solver-jacobi")))]
compile_error!("Enable exactly one of `solver-jacobi` (default) or `solver-gauss-seidel`.");

pub mod cloth_compute;
#[cfg(feature = "solver-jacobi")]
pub mod cloth_jacobi;
pub mod cloth_material;
pub mod mesh_prep;
pub mod xpbd_cpu;
