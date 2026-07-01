//! CPU reference for the GPU XPBD loop in `cloth_sim.wgsl` (optional self-collision; no normals).
//! Uses the same `constraint_*` row order / batch coloring as [`crate::mesh_prep::ClothMeshData`]
//! (Gauss–Seidel: one in-place jac buffer; constraints processed in contiguous index order).

use bevy::math::Vec3;

use crate::cloth_compute::{
    COLLISION_APPLY_CLAMP, JACOBI_CORRECTION_CAP, PREDICT_MAX_SPEED, SUBSTEPS,
};

/// Uniform inputs needed for predict + GS constraints (collision stripped).
#[derive(Clone, Debug)]
pub struct XpbdCpuTimeStepParams {
    pub dt: f32,
    pub inv_dt: f32,
    pub jacobi_omega: f32,
    pub inner_iterations: u32,
    /// Gravity in `predict` (`v += gravity * dt`), matching `cloth_sim.wgsl` (`params.gravity.xyz`).
    pub gravity: Vec3,
    pub floor_y: f32,
    pub grab_idx: i32,
    pub grab_active: bool,
    pub grab_target: Vec3,
    pub grab_stiffness: f32,
    /// `dv/dt` drag coefficient [1/s] matching WGSL `post_velocity`: `v *= exp(-linear_drag_per_sec * dt_sub)`.
    pub linear_drag_per_sec: f32,
}

fn clamp_correction(dx: Vec3) -> Vec3 {
    let ml = dx.length();
    let cap = JACOBI_CORRECTION_CAP;
    if ml > cap && ml > 0.0 {
        dx * (cap / ml)
    } else {
        dx
    }
}

fn predict_particle(
    i: usize,
    sim: Vec3,
    vel: Vec3,
    inv_mass: f32,
    p: &XpbdCpuTimeStepParams,
) -> (Vec3, Vec3) {
    if inv_mass <= 0.0 {
        return (sim, vel);
    }
    let mut v = vel;
    v += p.gravity * p.dt;
    let speed = v.length();
    let max_v = PREDICT_MAX_SPEED;
    if speed > max_v && speed > 0.0 {
        v *= max_v / speed;
    }
    let mut pos = sim + v * p.dt;
    if pos.y < p.floor_y {
        pos.y = p.floor_y;
    }
    if p.grab_active && i as i32 == p.grab_idx {
        let to_t = p.grab_target - pos;
        pos += to_t * p.grab_stiffness;
    }
    (pos, v)
}

/// O(n²) self-collision pass matching the GPU narrow phase in **`collide_grid_cells`** + **`collide_apply`** (`cloth_sim.wgsl`).
/// (upper triangle pairs, rest-aware folding test, mass-weighted split, per-axis displacement clamp).
pub fn self_collision_resolve(
    sim_pos: &mut [Vec3],
    inv_mass: &[f32],
    rest_pos: &[Vec3],
    thickness: f32,
    coll_scale: f32,
) {
    let n = sim_pos.len();
    debug_assert_eq!(inv_mass.len(), n);
    debug_assert_eq!(rest_pos.len(), n);
    let thickness_sq = thickness * thickness;

    let mut accum = vec![Vec3::ZERO; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if inv_mass[i] <= 0.0 && inv_mass[j] <= 0.0 {
                continue;
            }
            let p_i = sim_pos[i];
            let p_j = sim_pos[j];
            let mut d = p_j - p_i;
            let dist2 = d.length_squared();
            if dist2 > thickness_sq || dist2 < 1e-18 {
                continue;
            }
            let r0 = rest_pos[i];
            let r1 = rest_pos[j];
            let rest_d = r1 - r0;
            let rest2 = rest_d.length_squared();
            if dist2 > rest2 {
                continue;
            }
            let mut min_d = thickness;
            if rest2 < thickness_sq {
                min_d = rest2.sqrt();
            }
            let dist = dist2.sqrt();
            let corr = (min_d - dist) * 0.5 * coll_scale;
            if corr <= 0.0 {
                continue;
            }
            d = (d / dist) * corr;
            let w_i = inv_mass[i];
            let w_j = inv_mass[j];
            let inv_w = 1.0 / (w_i + w_j).max(1e-8);
            let di = -d * w_i * inv_w;
            let dj = d * w_j * inv_w;
            if w_i > 0.0 {
                accum[i] += di;
            }
            if w_j > 0.0 {
                accum[j] += dj;
            }
        }
    }

    let max_d = COLLISION_APPLY_CLAMP;
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            continue;
        }
        let c = accum[i];
        sim_pos[i] += Vec3::new(
            c.x.clamp(-max_d, max_d),
            c.y.clamp(-max_d, max_d),
            c.z.clamp(-max_d, max_d),
        );
    }
}

fn xpbd_substep_integrate(
    sim_pos: &mut [Vec3],
    prev_pos: &mut [Vec3],
    vel: &mut [Vec3],
    jac_work: &mut [Vec3],
    inv_mass: &[f32],
    constraint_i: &[u32],
    constraint_j: &[u32],
    constraint_rest_len: &[f32],
    constraint_compliance: &[f32],
    p: &XpbdCpuTimeStepParams,
) {
    let n = sim_pos.len();
    debug_assert_eq!(prev_pos.len(), n);
    debug_assert_eq!(vel.len(), n);
    debug_assert_eq!(jac_work.len(), n);
    debug_assert_eq!(inv_mass.len(), n);

    let num_constraints = constraint_i.len();
    let mut lambda = vec![0f32; num_constraints];

    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            prev_pos[i] = sim_pos[i];
            continue;
        }
        let (new_pos, new_v) = predict_particle(i, sim_pos[i], vel[i], inv_mass[i], p);
        prev_pos[i] = sim_pos[i];
        sim_pos[i] = new_pos;
        vel[i] = new_v;
    }

    for i in 0..n {
        jac_work[i] = sim_pos[i];
    }

    for _k in 0..p.inner_iterations {
        lambda.fill(0.0);
        debug_assert_eq!(constraint_j.len(), num_constraints);
        debug_assert_eq!(constraint_rest_len.len(), num_constraints);
        debug_assert_eq!(constraint_compliance.len(), num_constraints);

        let omega = p.jacobi_omega;
        let dt = p.dt;

        for e in 0..num_constraints {
            let i = constraint_i[e] as usize;
            let j = constraint_j[e] as usize;
            let w_i = inv_mass[i];
            let w_j = inv_mass[j];
            if w_j <= 0.0 && w_i <= 0.0 {
                continue;
            }
            let rest = constraint_rest_len[e];
            let compliance = constraint_compliance[e];
            let p_i = jac_work[i];
            let p_j = jac_work[j];
            let mut gv = p_i - p_j;
            let len = gv.length();
            if len < 1e-8 {
                continue;
            }
            gv /= len;
            let c = len - rest;
            let alpha_t = compliance / (dt * dt);
            let wsum = w_i + w_j + alpha_t;
            if wsum < 1e-8 {
                continue;
            }
            let lam = lambda[e];
            let dlam = (-c - alpha_t * lam) / wsum;
            lambda[e] = lam + dlam;

            let dx_i = clamp_correction(omega * gv * w_i * dlam);
            let dx_j = clamp_correction(-(omega * gv * w_j * dlam));
            if w_i > 0.0 {
                jac_work[i] = p_i + dx_i;
            }
            if w_j > 0.0 {
                jac_work[j] = p_j + dx_j;
            }
        }
    }

    for i in 0..n {
        sim_pos[i] = jac_work[i];
    }
}

fn xpbd_substep_post_velocity(
    sim_pos: &[Vec3],
    prev_pos: &[Vec3],
    vel: &mut [Vec3],
    inv_mass: &[f32],
    inv_dt: f32,
    dt_sub: f32,
    linear_drag_per_sec: f32,
) {
    let damp = (-linear_drag_per_sec * dt_sub).exp();
    for i in 0..sim_pos.len() {
        if inv_mass[i] <= 0.0 {
            continue;
        }
        vel[i] = (sim_pos[i] - prev_pos[i]) * inv_dt * damp;
    }
}

/// One GPU **substep** (predict → GS × K → sim ← jac → post_velocity), no collision.
pub fn xpbd_substep_no_collision(
    sim_pos: &mut [Vec3],
    prev_pos: &mut [Vec3],
    vel: &mut [Vec3],
    jac_work: &mut [Vec3],
    inv_mass: &[f32],
    constraint_i: &[u32],
    constraint_j: &[u32],
    constraint_rest_len: &[f32],
    constraint_compliance: &[f32],
    p: &XpbdCpuTimeStepParams,
) {
    xpbd_substep_integrate(
        sim_pos,
        prev_pos,
        vel,
        jac_work,
        inv_mass,
        constraint_i,
        constraint_j,
        constraint_rest_len,
        constraint_compliance,
        p,
    );
    xpbd_substep_post_velocity(
        sim_pos,
        prev_pos,
        vel,
        inv_mass,
        p.inv_dt,
        p.dt,
        p.linear_drag_per_sec,
    );
}

/// Like [`xpbd_substep_no_collision`], but runs [`self_collision_resolve`] after constraints (same order as the render graph).
pub fn xpbd_substep_with_self_collision(
    sim_pos: &mut [Vec3],
    prev_pos: &mut [Vec3],
    vel: &mut [Vec3],
    jac_work: &mut [Vec3],
    inv_mass: &[f32],
    constraint_i: &[u32],
    constraint_j: &[u32],
    constraint_rest_len: &[f32],
    constraint_compliance: &[f32],
    rest_pos: &[Vec3],
    thickness: f32,
    coll_scale: f32,
    p: &XpbdCpuTimeStepParams,
) {
    debug_assert_eq!(rest_pos.len(), sim_pos.len());
    xpbd_substep_integrate(
        sim_pos,
        prev_pos,
        vel,
        jac_work,
        inv_mass,
        constraint_i,
        constraint_j,
        constraint_rest_len,
        constraint_compliance,
        p,
    );
    self_collision_resolve(sim_pos, inv_mass, rest_pos, thickness, coll_scale);
    xpbd_substep_post_velocity(
        sim_pos,
        prev_pos,
        vel,
        inv_mass,
        p.inv_dt,
        p.dt,
        p.linear_drag_per_sec,
    );
}

/// One GPU **frame**: `SUBSTEPS` substeps (collision still omitted).
pub fn xpbd_frame_no_collision(
    sim_pos: &mut [Vec3],
    prev_pos: &mut [Vec3],
    vel: &mut [Vec3],
    jac_work: &mut [Vec3],
    inv_mass: &[f32],
    constraint_i: &[u32],
    constraint_j: &[u32],
    constraint_rest_len: &[f32],
    constraint_compliance: &[f32],
    substep: &XpbdCpuTimeStepParams,
) {
    for _ in 0..SUBSTEPS {
        xpbd_substep_no_collision(
            sim_pos,
            prev_pos,
            vel,
            jac_work,
            inv_mass,
            constraint_i,
            constraint_j,
            constraint_rest_len,
            constraint_compliance,
            substep,
        );
    }
}

/// One GPU **frame** with self-collision each substep (`SUBSTEPS` × integrate → collision → velocity).
pub fn xpbd_frame_with_self_collision(
    sim_pos: &mut [Vec3],
    prev_pos: &mut [Vec3],
    vel: &mut [Vec3],
    jac_work: &mut [Vec3],
    inv_mass: &[f32],
    constraint_i: &[u32],
    constraint_j: &[u32],
    constraint_rest_len: &[f32],
    constraint_compliance: &[f32],
    rest_pos: &[Vec3],
    thickness: f32,
    coll_scale: f32,
    substep: &XpbdCpuTimeStepParams,
) {
    for _ in 0..SUBSTEPS {
        xpbd_substep_with_self_collision(
            sim_pos,
            prev_pos,
            vel,
            jac_work,
            inv_mass,
            constraint_i,
            constraint_j,
            constraint_rest_len,
            constraint_compliance,
            rest_pos,
            thickness,
            coll_scale,
            substep,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth_compute::{ClothSimUniforms, INNER_ITERS, REFERENCE_FRAME_DELTA_SECS};
    use crate::mesh_prep::{grid_cloth_hanging, parse_welded_obj, ClothMeshData};
    use bevy::math::Vec4Swizzles;

    fn cloth_test_grid() -> ClothMeshData {
        grid_cloth_hanging(16, 16, 0.045)
    }

    #[test]
    fn cloth_mesh_inv_mass_and_neighbors_sane() {
        let cloth = cloth_test_grid();
        for (i, &w) in cloth.inv_mass.iter().enumerate() {
            assert!(w.is_finite(), "inv_mass[{}] = {} (non-finite)", i, w);
        }
    }

    /// Integrator only (`jacobi_omega = 0`): constraint pass is a no-op; catches NaNs in predict / floor / grab.
    #[test]
    fn cpu_xpbd_cloth_one_frame_integrator_only_stays_finite() {
        let cloth = cloth_test_grid();
        let n = cloth.num_particles as usize;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;

        let mut sim_pos: Vec<Vec3> = cloth.positions.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: 0.0,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: u.grab_idx,
            grab_active: u.grab_active != 0,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        xpbd_frame_no_collision(
            &mut sim_pos,
            &mut prev_pos,
            &mut vel,
            &mut jac_work,
            &cloth.inv_mass,
            &cloth.constraint_i,
            &cloth.constraint_j,
            &cloth.constraint_rest_len,
            &cloth.constraint_compliance,
            &sub,
        );

        assert!(sim_pos.iter().all(|p| p.is_finite()));
    }

    /// Full GPU-matching substep count + GS constraints on real cloth data (no collision).
    #[test]
    fn cpu_xpbd_cloth_one_frame_full_solver_stays_finite() {
        let cloth = cloth_test_grid();
        let n = cloth.num_particles as usize;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;

        let mut sim_pos: Vec<Vec3> = cloth.positions.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: u.grab_idx,
            grab_active: u.grab_active != 0,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        xpbd_frame_no_collision(
            &mut sim_pos,
            &mut prev_pos,
            &mut vel,
            &mut jac_work,
            &cloth.inv_mass,
            &cloth.constraint_i,
            &cloth.constraint_j,
            &cloth.constraint_rest_len,
            &cloth.constraint_compliance,
            &sub,
        );

        assert!(
            sim_pos.iter().all(|p| p.is_finite()),
            "expected finite positions after 1 frame CPU XPBD (matches WGSL predict gravity)"
        );
    }

    #[test]
    fn cpu_xpbd_pins_hold_rest_positions() {
        let cloth = cloth_test_grid();
        let n = cloth.num_particles as usize;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;

        let rest = cloth.positions.clone();
        let mut sim_pos = rest.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: -1,
            grab_active: false,
            grab_target: Vec3::ZERO,
            grab_stiffness: 0.0,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        xpbd_frame_no_collision(
            &mut sim_pos,
            &mut prev_pos,
            &mut vel,
            &mut jac_work,
            &cloth.inv_mass,
            &cloth.constraint_i,
            &cloth.constraint_j,
            &cloth.constraint_rest_len,
            &cloth.constraint_compliance,
            &sub,
        );

        let eps = 1e-4_f32;
        for i in 0..n {
            if cloth.inv_mass[i] <= 0.0 {
                assert!(
                    (sim_pos[i] - rest[i]).length() < eps,
                    "pinned particle {} moved: {:?} vs {:?}",
                    i,
                    sim_pos[i],
                    rest[i]
                );
            }
        }
    }

    /// Single triangle, three UV corners — sanity check independent of `cloth.obj` size.
    #[test]
    fn cpu_xpbd_single_triangle_one_frame_finite() {
        let obj = r#"
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"#;
        let cloth = parse_welded_obj(obj);
        assert_eq!(cloth.num_particles, 3);
        let n = 3;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;

        let mut sim_pos = cloth.positions.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: Vec3::new(-1.0, -2.0, 0.5),
            floor_y: -10.0,
            grab_idx: -1,
            grab_active: false,
            grab_target: Vec3::ZERO,
            grab_stiffness: 0.0,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        xpbd_frame_no_collision(
            &mut sim_pos,
            &mut prev_pos,
            &mut vel,
            &mut jac_work,
            &cloth.inv_mass,
            &cloth.constraint_i,
            &cloth.constraint_j,
            &cloth.constraint_rest_len,
            &cloth.constraint_compliance,
            &sub,
        );

        for (i, p) in sim_pos.iter().enumerate() {
            assert!(p.is_finite(), "tri particle {} = {:?}", i, p);
        }
    }

    /// Regression: many frames without self-collision should not collapse free vertices to a near-degenerate AABB.
    #[test]
    fn cpu_xpbd_cloth_15_frames_extent_stays_reasonable() {
        let cloth = cloth_test_grid();
        let n = cloth.num_particles as usize;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;

        let mut sim_pos: Vec<Vec3> = cloth.positions.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: -1,
            grab_active: false,
            grab_target: Vec3::ZERO,
            grab_stiffness: 0.0,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        for _ in 0..15 {
            xpbd_frame_no_collision(
                &mut sim_pos,
                &mut prev_pos,
                &mut vel,
                &mut jac_work,
                &cloth.inv_mass,
                &cloth.constraint_i,
                &cloth.constraint_j,
                &cloth.constraint_rest_len,
                &cloth.constraint_compliance,
                &sub,
            );
        }

        let mut min_b = Vec3::splat(f32::INFINITY);
        let mut max_b = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..n {
            if cloth.inv_mass[i] > 0.0 {
                min_b = min_b.min(sim_pos[i]);
                max_b = max_b.max(sim_pos[i]);
            }
        }
        let ext = max_b - min_b;
        assert!(
            ext.x > 0.18 && ext.y > 0.04,
            "free-particle AABB nearly degenerate in XY (collapse / ball): {:?}",
            ext
        );
    }

    #[test]
    fn cpu_self_collision_pair_separates_when_penetrating() {
        let thickness = 0.04_f32;
        let coll_scale = 0.38_f32;
        let inv_mass = vec![1.0_f32, 1.0];
        let rest_pos = vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let mut sim = vec![Vec3::ZERO, Vec3::new(0.015, 0.0, 0.0)];
        let d0 = (sim[1] - sim[0]).length();
        self_collision_resolve(&mut sim, &inv_mass, &rest_pos, thickness, coll_scale);
        let d1 = (sim[1] - sim[0]).length();
        assert!(
            d1 > d0 + 1e-4,
            "expected separation to grow: {} -> {}",
            d0,
            d1
        );
    }

    #[test]
    fn cpu_self_collision_skips_when_world_separation_exceeds_rest() {
        let thickness = 0.04_f32;
        let coll_scale = 0.38_f32;
        let inv_mass = vec![1.0_f32, 1.0];
        let rest_pos = vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let mut sim = vec![Vec3::ZERO, Vec3::new(0.15, 0.0, 0.0)];
        let before = sim.clone();
        self_collision_resolve(&mut sim, &inv_mass, &rest_pos, thickness, coll_scale);
        assert_eq!(sim[0], before[0]);
        assert_eq!(sim[1], before[1]);
    }

    /// WGSL skips `atomicAdd` for zero `inv_mass`; pinned vertex should not move from self-collision.
    #[test]
    fn cpu_self_collision_pinned_neighbor_takes_no_delta() {
        let thickness = 0.04_f32;
        let coll_scale = 0.38_f32;
        let inv_mass = vec![0.0_f32, 1.0];
        let rest_pos = vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let mut sim = vec![Vec3::ZERO, Vec3::new(0.015, 0.0, 0.0)];
        let pin = sim[0];
        self_collision_resolve(&mut sim, &inv_mass, &rest_pos, thickness, coll_scale);
        assert_eq!(sim[0], pin);
        assert!(
            sim[1].x > 0.015,
            "free particle should separate: {:?}",
            sim[1]
        );
    }

    #[test]
    fn cpu_xpbd_cloth_one_frame_with_self_collision_stays_finite() {
        let cloth = cloth_test_grid();
        let n = cloth.num_particles as usize;
        let u = ClothSimUniforms::default();
        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;
        let rest = cloth.positions.clone();

        let mut sim_pos: Vec<Vec3> = rest.clone();
        let mut prev_pos = vec![Vec3::ZERO; n];
        let mut vel = vec![Vec3::ZERO; n];
        let mut jac_work = vec![Vec3::ZERO; n];

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: u.grab_idx,
            grab_active: u.grab_active != 0,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        xpbd_frame_with_self_collision(
            &mut sim_pos,
            &mut prev_pos,
            &mut vel,
            &mut jac_work,
            &cloth.inv_mass,
            &cloth.constraint_i,
            &cloth.constraint_j,
            &cloth.constraint_rest_len,
            &cloth.constraint_compliance,
            &rest,
            u.thickness,
            u.coll_scale,
            &sub,
        );

        assert!(
            sim_pos.iter().all(|p| p.is_finite()),
            "expected finite positions after 1 frame with CPU self-collision"
        );
    }
}
