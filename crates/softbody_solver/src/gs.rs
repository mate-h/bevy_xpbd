//! Gauss–Seidel solver entry points — bindings match host GS layout (group 0, 0–27).

use spirv_std::glam::{UVec3, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

use crate::atom;
use crate::common::{
    clamp_delta_vec, collision_flat_packed, effective_inv_mass, neighbor_flat, predict_position,
    self_collision_separation, xpbd_distance_delta_lambda, xyz,
};
use crate::types::{
    CollGridUniform, CollRadixPassUniform, FIXSCALE, GsDynBatchUniform, NORM_SCALE, SimParams,
};

#[inline]
fn perm_read(ping: &[u32], pong: &[u32], idx: usize, from_pong: bool) -> u32 {
    if from_pong {
        pong[idx]
    } else {
        ping[idx]
    }
}

#[inline]
fn perm_final_read(ping: &[u32], pong: &[u32], i: usize, radix_digits: u32) -> u32 {
    let out_ping = (radix_digits & 1) == 0;
    perm_read(ping, pong, i, !out_ping)
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn predict_copy_sim_to_jac(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jac_state: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] prev_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] velocities: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] inv_mass: &[f32],
) {
    let i = gid.x as usize;
    if i >= params.num_particles as usize {
        return;
    }
    let w = inv_mass[i];
    if w <= 0.0 {
        prev_pos[i] = sim_pos[i];
        jac_state[i] = sim_pos[i];
        return;
    }
    prev_pos[i] = sim_pos[i];
    let (p, v) = predict_position(params, gid.x, sim_pos[i], velocities[i], w);
    sim_pos[i] = p;
    velocities[i] = v;
    jac_state[i] = p;
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn copy_jac_to_sim(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jac_state: &mut [Vec4],
) {
    let i = gid.x as usize;
    if i >= params.num_particles as usize {
        return;
    }
    sim_pos[i] = jac_state[i];
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn clear_constraint_lambda(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] constraint_lambda: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 13)] constraint_delta_lambda: &mut [f32],
) {
    let i = gid.x as usize;
    if i >= constraint_lambda.len() {
        return;
    }
    constraint_lambda[i] = 0.0;
    constraint_delta_lambda[i] = 0.0;
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(128)))]
pub fn gs_edges(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(uniform, descriptor_set = 0, binding = 19)] gs_dyn_batch: &GsDynBatchUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jac_state: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] inv_mass: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] constraint_batch_offsets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] constraint_i: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] constraint_j: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] constraint_rest: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] constraint_comp: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] constraint_lambda: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 13)] constraint_delta_lambda: &mut [f32],
) {
    let b = gs_dyn_batch.head.x;
    if params.constraint_batch_count == 0 || b >= params.constraint_batch_count {
        return;
    }
    let start = constraint_batch_offsets[b as usize];
    let end = constraint_batch_offsets[b as usize + 1];
    let e = gid.x + start;
    if e >= end {
        return;
    }
    let e = e as usize;
    let i = constraint_i[e] as usize;
    let j = constraint_j[e] as usize;
    let w_i = effective_inv_mass(params, i as u32, inv_mass[i]);
    let w_j = effective_inv_mass(params, j as u32, inv_mass[j]);
    if w_j <= 0.0 && w_i <= 0.0 {
        constraint_delta_lambda[e] = 0.0;
        return;
    }
    let rest = constraint_rest[e];
    let compliance = constraint_comp[e];
    let p_i = xyz(jac_state[i]);
    let p_j = xyz(jac_state[j]);
    let mut gv = p_i - p_j;
    let len = gv.length();
    let lambda_e = constraint_lambda[e];
    let dlam = xpbd_distance_delta_lambda(
        len,
        rest,
        w_i,
        w_j,
        compliance,
        params.inv_dt_sq,
        lambda_e,
    );
    if dlam == 0.0 {
        constraint_delta_lambda[e] = 0.0;
        return;
    }
    gv /= len;
    constraint_delta_lambda[e] = dlam;
    constraint_lambda[e] = lambda_e + dlam;
    let omega = params.jacobi_omega;
    let mut dx_i = omega * gv * w_i * dlam;
    let mut dx_j = -(omega * gv * w_j * dlam);
    dx_i = clamp_delta_vec(dx_i);
    dx_j = clamp_delta_vec(dx_j);
    if w_i > 0.0 {
        let p = p_i + dx_i;
        jac_state[i] = Vec4::new(p.x, p.y, p.z, 0.0);
    }
    if w_j > 0.0 {
        let p = p_j + dx_j;
        jac_state[j] = Vec4::new(p.x, p.y, p.z, 0.0);
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn post_velocity(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] prev_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] velocities: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] inv_mass: &[f32],
) {
    let i = gid.x as usize;
    if i >= params.num_particles as usize || inv_mass[i] <= 0.0 {
        return;
    }
    let mut v = (xyz(sim_pos[i]) - xyz(prev_pos[i])) * params.inv_dt;
    let damp = libm::expf(-params.linear_drag_per_sec * params.dt);
    v *= damp;
    velocities[i] = Vec4::new(v.x, v.y, v.z, 0.0);
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn clear_atomics(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 17)] atomic_coll: &mut [i32],
) {
    let i = gid.x as usize;
    if i < (params.num_particles * 3) as usize {
        unsafe { atom::store_i32(&mut atomic_coll[i], 0) }
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_cell_bounds_clear(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 26)] coll_cell_start: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 27)] coll_cell_end_exclusive: &mut [u32],
) {
    let i = gid.x as usize;
    if i >= coll_grid_u.num_cells as usize {
        return;
    }
    unsafe {
        atom::store_u32(&mut coll_cell_start[i], 0xffff_ffff);
        atom::store_u32(&mut coll_cell_end_exclusive[i], 0);
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_perm_identity_ping(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 24)] coll_perm_ping: &mut [u32],
) {
    let i = gid.x as usize;
    if i >= coll_grid_u.num_particles as usize {
        return;
    }
    coll_perm_ping[i] = gid.x;
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_histogram_clear(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 22)] coll_radix_hist: &mut [u32; 256],
) {
    let i = gid.x as usize;
    if i >= 256 {
        return;
    }
    unsafe { atom::store_u32(&mut coll_radix_hist[i], 0) }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_radix_digit_count(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(uniform, descriptor_set = 0, binding = 21)] coll_radix_pass_u: &CollRadixPassUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 22)] coll_radix_hist: &mut [u32; 256],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 24)] coll_perm_ping: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 25)] coll_perm_pong: &mut [u32],
) {
    let in_idx = gid.x as usize;
    if in_idx >= coll_grid_u.num_particles as usize {
        return;
    }
    let rd_pass = coll_radix_pass_u.data.x;
    let pj = perm_read(coll_perm_ping, coll_perm_pong, in_idx, (rd_pass & 1) != 0);
    let f = collision_flat_packed(coll_grid_u, xyz(sim_pos[pj as usize]));
    let digit = ((f >> (rd_pass * 8)) & 0xff) as usize;
    unsafe {
        atom::add_u32(&mut coll_radix_hist[digit], 1);
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(1)))]
pub fn coll_radix_exclusive_bases_heads(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 22)] coll_radix_hist: &mut [u32; 256],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 23)] coll_radix_head: &mut [u32; 256],
) {
    if gid.x != 0 || gid.y != 0 || gid.z != 0 {
        return;
    }
    let mut acc = 0u32;
    let mut d = 0usize;
    while d < 256 {
        let c = unsafe { atom::load_u32(&coll_radix_hist[d]) };
        unsafe { atom::store_u32(&mut coll_radix_head[d], acc) };
        acc += c;
        d += 1;
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_radix_digit_scatter(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(uniform, descriptor_set = 0, binding = 21)] coll_radix_pass_u: &CollRadixPassUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 23)] coll_radix_head: &mut [u32; 256],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 24)] coll_perm_ping: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 25)] coll_perm_pong: &mut [u32],
) {
    let in_idx = gid.x as usize;
    if in_idx >= coll_grid_u.num_particles as usize {
        return;
    }
    let rd_pass = coll_radix_pass_u.data.x;
    let from_pong = (rd_pass & 1) != 0;
    let pj = if from_pong {
        coll_perm_pong[in_idx]
    } else {
        coll_perm_ping[in_idx]
    };
    let f = collision_flat_packed(coll_grid_u, xyz(sim_pos[pj as usize]));
    let digit = ((f >> (rd_pass * 8)) & 0xff) as usize;
    let tgt = unsafe { atom::add_u32(&mut coll_radix_head[digit], 1) } as usize;
    if (rd_pass & 1) != 0 {
        coll_perm_ping[tgt] = pj;
    } else {
        coll_perm_pong[tgt] = pj;
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn coll_sorted_build_cell_ranges(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 24)] coll_perm_ping: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 25)] coll_perm_pong: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 26)] coll_cell_start: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 27)] coll_cell_end_exclusive: &mut [u32],
) {
    let i = gid.x as usize;
    let np = coll_grid_u.num_particles as usize;
    if i >= np {
        return;
    }
    let cell = |idx: usize| -> u32 {
        let pj = perm_final_read(coll_perm_ping, coll_perm_pong, idx, coll_grid_u.radix_digits);
        collision_flat_packed(coll_grid_u, xyz(sim_pos[pj as usize]))
    };
    let cid_here = cell(i);
    let mut run_start = i == 0;
    if !run_start {
        run_start = cell(i - 1) != cid_here;
    }
    let mut run_end = i == np - 1;
    if !run_end {
        run_end = cell(i + 1) != cid_here;
    }
    if run_start {
        unsafe { atom::store_u32(&mut coll_cell_start[cid_here as usize], gid.x) }
    }
    if run_end {
        unsafe { atom::store_u32(&mut coll_cell_end_exclusive[cid_here as usize], gid.x + 1) }
    }
}

fn narrow_self_collision_pair(
    params: &SimParams,
    sim_pos: &mut [Vec4],
    rest_pos: &[Vec4],
    inv_mass: &[f32],
    atomic_coll: &mut [i32],
    i: usize,
    j: usize,
) {
    let w_i = effective_inv_mass(params, i as u32, inv_mass[i]);
    let w_j = effective_inv_mass(params, j as u32, inv_mass[j]);
    let (hit, di, dj) = self_collision_separation(
        params.thickness,
        params.coll_scale,
        xyz(sim_pos[i]),
        xyz(sim_pos[j]),
        xyz(rest_pos[i]),
        xyz(rest_pos[j]),
        w_i,
        w_j,
    );
    if !hit {
        return;
    }
    let fs = FIXSCALE as f32;
    unsafe {
        if w_i > 0.0 {
            atom::add_i32(&mut atomic_coll[i * 3], (di.x * fs) as i32);
            atom::add_i32(&mut atomic_coll[i * 3 + 1], (di.y * fs) as i32);
            atom::add_i32(&mut atomic_coll[i * 3 + 2], (di.z * fs) as i32);
        }
        if w_j > 0.0 {
            atom::add_i32(&mut atomic_coll[j * 3], (dj.x * fs) as i32);
            atom::add_i32(&mut atomic_coll[j * 3 + 1], (dj.y * fs) as i32);
            atom::add_i32(&mut atomic_coll[j * 3 + 2], (dj.z * fs) as i32);
        }
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn collide_grid_cells(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(uniform, descriptor_set = 0, binding = 20)] coll_grid_u: &CollGridUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] rest_pos: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] inv_mass: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 17)] atomic_coll: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 24)] coll_perm_ping: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 25)] coll_perm_pong: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 26)] coll_cell_start: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 27)] coll_cell_end_exclusive: &mut [u32],
) {
    if params.coll_scale <= 0.0 {
        return;
    }
    let i = gid.x as usize;
    if i >= params.num_particles as usize {
        return;
    }
    let my_cell = collision_flat_packed(coll_grid_u, xyz(sim_pos[i]));
    let mut dz = -1i32;
    while dz <= 1 {
        let mut dy = -1i32;
        while dy <= 1 {
            let mut dx = -1i32;
            while dx <= 1 {
                let nf = neighbor_flat(coll_grid_u, my_cell, dx, dy, dz);
                if nf < coll_grid_u.num_cells {
                    let s = unsafe { atom::load_u32(&coll_cell_start[nf as usize]) };
                    let e = unsafe { atom::load_u32(&coll_cell_end_exclusive[nf as usize]) };
                    if s != 0xffff_ffff {
                        let mut idx = s;
                        while idx < e {
                            let j = perm_final_read(
                                coll_perm_ping,
                                coll_perm_pong,
                                idx as usize,
                                coll_grid_u.radix_digits,
                            );
                            if j > gid.x {
                                narrow_self_collision_pair(
                                    params,
                                    sim_pos,
                                    rest_pos,
                                    inv_mass,
                                    atomic_coll,
                                    i,
                                    j as usize,
                                );
                            }
                            idx += 1;
                        }
                    }
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn collide_apply(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] inv_mass: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 17)] atomic_coll: &mut [i32],
) {
    let i = gid.x as usize;
    if i >= params.num_particles as usize
        || effective_inv_mass(params, gid.x, inv_mass[i]) <= 0.0
    {
        return;
    }
    let fs = FIXSCALE as f32;
    let mut sx = unsafe { atom::load_i32(&atomic_coll[i * 3]) } as f32 / fs;
    let mut sy = unsafe { atom::load_i32(&atomic_coll[i * 3 + 1]) } as f32 / fs;
    let mut sz = unsafe { atom::load_i32(&atomic_coll[i * 3 + 2]) } as f32 / fs;
    let max_d = 0.35;
    sx = sx.clamp(-max_d, max_d);
    sy = sy.clamp(-max_d, max_d);
    sz = sz.clamp(-max_d, max_d);
    let p = xyz(sim_pos[i]) + Vec3::new(sx, sy, sz);
    sim_pos[i] = Vec4::new(p.x, p.y, p.z, 0.0);
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(256)))]
pub fn clear_norm_atomics(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 18)] atomic_norm: &mut [i32],
) {
    let i = gid.x as usize;
    if i < (params.num_particles * 3) as usize {
        unsafe { atom::store_i32(&mut atomic_norm[i], 0) }
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn accumulate_normals(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 14)] tri_indices: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 18)] atomic_norm: &mut [i32],
) {
    let t = gid.x as usize;
    if t >= params.num_tris as usize {
        return;
    }
    let i0 = tri_indices[t * 3] as usize;
    let i1 = tri_indices[t * 3 + 1] as usize;
    let i2 = tri_indices[t * 3 + 2] as usize;
    let c = (xyz(sim_pos[i1]) - xyz(sim_pos[i0])).cross(xyz(sim_pos[i2]) - xyz(sim_pos[i0]));
    let f = NORM_SCALE;
    unsafe {
        atom::add_i32(&mut atomic_norm[i0 * 3], (c.x * f) as i32);
        atom::add_i32(&mut atomic_norm[i0 * 3 + 1], (c.y * f) as i32);
        atom::add_i32(&mut atomic_norm[i0 * 3 + 2], (c.z * f) as i32);
        atom::add_i32(&mut atomic_norm[i1 * 3], (c.x * f) as i32);
        atom::add_i32(&mut atomic_norm[i1 * 3 + 1], (c.y * f) as i32);
        atom::add_i32(&mut atomic_norm[i1 * 3 + 2], (c.z * f) as i32);
        atom::add_i32(&mut atomic_norm[i2 * 3], (c.x * f) as i32);
        atom::add_i32(&mut atomic_norm[i2 * 3 + 1], (c.y * f) as i32);
        atom::add_i32(&mut atomic_norm[i2 * 3 + 2], (c.z * f) as i32);
    }
}

#[unsafe(no_mangle)]
#[spirv(compute(threads(64)))]
pub fn finalize_normals(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sim_pos: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 15)] render_positions: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 16)] render_normals: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 18)] atomic_norm: &mut [i32],
) {
    let i = gid.x as usize;
    if i >= params.num_particles as usize {
        return;
    }
    let f = NORM_SCALE;
    let mut n = Vec3::new(
        unsafe { atom::load_i32(&atomic_norm[i * 3]) } as f32 / f,
        unsafe { atom::load_i32(&atomic_norm[i * 3 + 1]) } as f32 / f,
        unsafe { atom::load_i32(&atomic_norm[i * 3 + 2]) } as f32 / f,
    );
    let ln = n.length();
    if ln > 1e-8 {
        n /= ln;
    } else {
        n = Vec3::new(0.0, 1.0, 0.0);
    }
    render_normals[i] = Vec4::new(n.x, n.y, n.z, 0.0);
    render_positions[i] = sim_pos[i];
}
