#[cfg(target_arch = "spirv")]
use spirv_std::glam::{Vec3, Vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
#[cfg(not(target_arch = "spirv"))]
use glam::{Vec3, Vec4};

use crate::types::{
    CollGridUniform, CORRECTION_CAP, EDGE_COMPRESS_MIN_FRAC, GRAB_MAX_PULL, PREDICT_MAX_SPEED,
    SimParams,
};

#[inline]
pub fn xyz(v: Vec4) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Grabbed particles are treated as kinematic (infinite mass) so XPBD solves *around*
/// the handle instead of fighting a predict-time spring yank.
#[inline]
pub fn is_grabbed(params: &SimParams, i: u32) -> bool {
    params.grab_active != 0 && i as i32 == params.grab_idx
}

#[inline]
pub fn effective_inv_mass(params: &SimParams, i: u32, inv_mass: f32) -> f32 {
    if is_grabbed(params, i) {
        0.0
    } else {
        inv_mass
    }
}

#[inline]
pub fn clamp_delta_vec(dx: Vec3) -> Vec3 {
    let ml = dx.length();
    if ml > CORRECTION_CAP && ml > 0.0 {
        dx * (CORRECTION_CAP / ml)
    } else {
        dx
    }
}

#[inline]
pub fn predict_position(
    params: &SimParams,
    i: u32,
    sim: Vec4,
    vel: Vec4,
    inv_mass: f32,
) -> (Vec4, Vec4) {
    // Kinematic grab: soft-follow the cursor, ignore gravity/integration for this particle.
    // Constraints see w=0 via [`effective_inv_mass`], so the rest of the cloth drapes from it.
    if is_grabbed(params, i) {
        let cur = xyz(sim);
        let target = xyz(params.grab_target);
        let mut pull = (target - cur) * params.grab_stiffness.clamp(0.0, 1.0);
        let pl = pull.length();
        if pl > GRAB_MAX_PULL && pl > 0.0 {
            pull *= GRAB_MAX_PULL / pl;
        }
        let p = cur + pull;
        return (
            Vec4::new(p.x, p.y, p.z, 0.0),
            Vec4::new(vel.x, vel.y, vel.z, 0.0),
        );
    }
    if inv_mass <= 0.0 {
        return (sim, vel);
    }
    let mut v = xyz(vel);
    v += xyz(params.gravity) * params.dt;
    let speed = v.length();
    let max_v = PREDICT_MAX_SPEED;
    if speed > max_v {
        v *= max_v / speed;
    }
    let p = xyz(sim) + v * params.dt;
    (
        Vec4::new(p.x, p.y, p.z, 0.0),
        Vec4::new(v.x, v.y, v.z, 0.0),
    )
}

/// XPBD distance Δλ (Macklin et al. Eq. 17–18): α̃ = α/Δt², Δλ = (−C − α̃λ) / (∑w + α̃).
///
/// When `len < rest * EDGE_COMPRESS_MIN_FRAC`, uses a **hard unilateral** expand toward that
/// floor (compliance ignored). Soft stretch alone cannot stop accordion furling; particle
/// self-collision also misses it because local pairs skip once `dist ≥ rest`.
#[inline]
pub fn xpbd_distance_delta_lambda(
    len: f32,
    rest: f32,
    w_i: f32,
    w_j: f32,
    compliance: f32,
    inv_dt_sq: f32,
    lambda: f32,
) -> f32 {
    if len < 1e-8 || (w_i <= 0.0 && w_j <= 0.0) {
        return 0.0;
    }
    let min_len = rest * EDGE_COMPRESS_MIN_FRAC;
    if len < min_len {
        let c = len - min_len;
        let wsum = w_i + w_j;
        if wsum < 1e-8 {
            return 0.0;
        }
        return (-c) / wsum;
    }
    let c = len - rest;
    let alpha_t = compliance * inv_dt_sq;
    let wsum = w_i + w_j + alpha_t;
    if wsum < 1e-8 {
        return 0.0;
    }
    (-c - alpha_t * lambda) / wsum
}

/// Particle–particle separation for self-collision.
/// Returns `(hit, di, dj)` — `hit == false` means no contact (`di`/`dj` are zero).
/// Uses a bool (not `Option`) so rust-gpu can compile the call sites.
///
/// - **Local** pairs (`rest < ~1.35 × thickness`): only when compressed below rest (don't inflate edges).
/// - **Non-local** pairs: always maintain a thickness shell — this is the layer/furl guard.
#[inline]
pub fn self_collision_separation(
    thickness: f32,
    coll_scale: f32,
    p_i: Vec3,
    p_j: Vec3,
    rest_i: Vec3,
    rest_j: Vec3,
    w_i: f32,
    w_j: f32,
) -> (bool, Vec3, Vec3) {
    let zero = Vec3::ZERO;
    if coll_scale <= 0.0 || (w_i <= 0.0 && w_j <= 0.0) {
        return (false, zero, zero);
    }
    let thickness_sq = thickness * thickness;
    let mut d = p_j - p_i;
    let dist2 = d.dot(d);
    if dist2 > thickness_sq {
        return (false, zero, zero);
    }
    let rest_d = rest_j - rest_i;
    let rest_len = rest_d.length();
    let local = rest_len < thickness * 1.35;
    if local && dist2 > rest_len * rest_len {
        return (false, zero, zero);
    }
    let min_d = if local {
        rest_len.min(thickness)
    } else {
        thickness
    };
    let dist = if dist2 > 1e-18 {
        dist2.sqrt()
    } else {
        0.0
    };
    let corr = (min_d - dist) * coll_scale;
    if corr <= 0.0 {
        return (false, zero, zero);
    }
    // Near-coincident contacts: direction from rest offset (world delta is noisy).
    if dist < 1e-5 {
        let rl = rest_len;
        if rl > 1e-8 {
            d = rest_d * (corr / rl);
        } else {
            d = Vec3::new(0.0, corr, 0.0);
        }
    } else {
        d *= corr / dist;
    }
    let inv_w = 1.0 / (w_i + w_j).max(1e-8);
    (true, -d * w_i * inv_w, d * w_j * inv_w)
}

#[inline]
pub fn collision_flat_packed(grid: &CollGridUniform, p: Vec3) -> u32 {
    let g = xyz(grid.grid_origin_pad);
    let q = (p - g) * grid.inv_cell;
    let gx = grid.gx as i32;
    let gy = grid.gy as i32;
    let gz = grid.gz as i32;
    let ix = q.x.floor() as i32;
    let iy = q.y.floor() as i32;
    let iz = q.z.floor() as i32;
    let ix = ix.clamp(0, gx - 1) as u32;
    let iy = iy.clamp(0, gy - 1) as u32;
    let iz = iz.clamp(0, gz - 1) as u32;
    ix + iy * grid.gx + iz * grid.gx * grid.gy
}

#[inline]
pub fn neighbor_flat(grid: &CollGridUniform, cid: u32, dx: i32, dy: i32, dz: i32) -> u32 {
    let gx = grid.gx;
    let gy = grid.gy;
    let gz = grid.gz;
    let iz = cid / (gx * gy);
    let t = cid - iz * gx * gy;
    let iy = t / gx;
    let ix = t - iy * gx;
    let nx = ix as i32 + dx;
    let ny = iy as i32 + dy;
    let nz = iz as i32 + dz;
    if nx < 0 || ny < 0 || nz < 0 || nx >= gx as i32 || ny >= gy as i32 || nz >= gz as i32 {
        return grid.num_cells;
    }
    nx as u32 + ny as u32 * gx + nz as u32 * gx * gy
}
