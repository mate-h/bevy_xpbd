//! GPU uniform layouts — must match `ClothSimParamsGpu` / friends in `bevy_softbody`.
//!
//! Use glam vector types (not `[T; N]` arrays) so SPIR-V Uniform Block layout matches
//! WGSL `vec2`/`vec4` and host `bytemuck` packing under standard UBO rules.

#[cfg(target_arch = "spirv")]
use spirv_std::glam::{UVec2, UVec4, Vec2, Vec4};
#[cfg(not(target_arch = "spirv"))]
use glam::{UVec2, UVec4, Vec2, Vec4};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SimParams {
    pub dt: f32,
    pub inv_dt: f32,
    pub inv_dt_sq: f32,
    pub constraint_batch_count: u32,
    pub num_particles: u32,
    pub num_tris: u32,
    pub jacobi_omega: f32,
    pub inner_iterations: u32,
    pub thickness: f32,
    pub coll_scale: f32,
    pub _pad_before_gravity: Vec2,
    pub gravity: Vec4,
    pub grab_target: Vec4,
    pub grab_idx: i32,
    pub grab_active: u32,
    pub grab_stiffness: f32,
    pub _pad_legacy_floor: f32,
    pub linear_drag_per_sec: f32,
    pub constraint_batch_idx: u32,
    pub _uniform_pad_vec2_u: UVec2,
    pub _uniform_pad_vec2_f: Vec2,
    pub _uniform_encase_reserve: UVec2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollGridUniform {
    pub grid_origin_pad: Vec4,
    pub inv_cell: f32,
    pub num_cells: u32,
    pub num_particles: u32,
    pub gx: u32,
    pub gy: u32,
    pub gz: u32,
    pub radix_digits: u32,
    pub _align_pad: u32,
    pub _reserved: UVec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollRadixPassUniform {
    pub data: UVec4,
}

/// 256-byte dynamic uniform slot for GS color batch index (`head.x`).
#[cfg(feature = "solver-gauss-seidel")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GsDynBatchUniform {
    pub head: UVec4,
    pub _pad_bulk: [UVec4; 15],
}

pub const FIXSCALE: i32 = 10000;
pub const CORRECTION_CAP: f32 = 0.28;
pub const GRAB_MAX_PULL: f32 = 0.028;
pub const NORM_SCALE: f32 = 33333.0;
pub const PREDICT_MAX_SPEED: f32 = 12.0;
/// Hard floor on distance-constraint length as a fraction of rest (anti-accordion / furl).
pub const EDGE_COMPRESS_MIN_FRAC: f32 = 0.70;
