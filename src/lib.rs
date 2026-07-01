//! GPU XPBD cloth simulation with Bevy 0.19 (WebGPU via wgpu).

pub mod cloth_compute;
pub mod cloth_material;
pub mod mesh_prep;
pub mod xpbd_cpu;

#[cfg(test)]
mod gpu_cpu_parity;
