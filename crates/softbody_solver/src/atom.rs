//! Thin wrappers around spirv-std atomics (QueueFamily scope, relaxed).
//!
//! QueueFamily matches typical WGSL storage-buffer atomics and avoids needing
//! `VulkanMemoryModelDeviceScopeKHR` (required for `Scope::Device`).

use spirv_std::arch::{atomic_i_add, atomic_load, atomic_store};
use spirv_std::memory::Scope;

pub const SCOPE: u32 = Scope::QueueFamily as u32;
pub const SEM: u32 = 0; // Semantics::NONE

#[inline]
pub unsafe fn store_i32(ptr: &mut i32, value: i32) {
    unsafe { atomic_store::<i32, SCOPE, SEM>(ptr, value) }
}

#[inline]
pub unsafe fn load_i32(ptr: &i32) -> i32 {
    unsafe { atomic_load::<i32, SCOPE, SEM>(ptr) }
}

#[inline]
pub unsafe fn add_i32(ptr: &mut i32, value: i32) -> i32 {
    unsafe { atomic_i_add::<i32, SCOPE, SEM>(ptr, value) }
}

#[inline]
pub unsafe fn store_u32(ptr: &mut u32, value: u32) {
    unsafe { atomic_store::<u32, SCOPE, SEM>(ptr, value) }
}

#[inline]
pub unsafe fn load_u32(ptr: &u32) -> u32 {
    unsafe { atomic_load::<u32, SCOPE, SEM>(ptr) }
}

#[inline]
pub unsafe fn add_u32(ptr: &mut u32, value: u32) -> u32 {
    unsafe { atomic_i_add::<u32, SCOPE, SEM>(ptr, value) }
}
