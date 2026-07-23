//! XPBD softbody compute kernels (rust-gpu).
//!
//! On `target_arch = "spirv"` this crate is the full solver (entry points in `jacobi` / `gs`).
//! On the host it exposes [`types`] and [`common`] so unit tests reuse the same math as the GPU path.

#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_arch = "spirv", deny(warnings))]

#[cfg(target_arch = "spirv")]
mod atom;
pub mod common;
pub mod types;

#[cfg(all(target_arch = "spirv", feature = "solver-jacobi"))]
mod jacobi;
#[cfg(all(target_arch = "spirv", feature = "solver-gauss-seidel"))]
mod gs;

#[cfg(all(target_arch = "spirv", feature = "solver-jacobi"))]
pub use jacobi::*;
#[cfg(all(target_arch = "spirv", feature = "solver-gauss-seidel"))]
pub use gs::*;

#[cfg(all(test, not(target_arch = "spirv")))]
mod tests;
