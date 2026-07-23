//! Host unit tests for shared XPBD / predict helpers (same code the GPU kernels call).

use glam::{UVec4, Vec2, Vec3, Vec4};

use crate::common::{
    clamp_delta_vec, collision_flat_packed, neighbor_flat, predict_position,
    xpbd_distance_delta_lambda, xyz,
};
use crate::types::{CollGridUniform, CORRECTION_CAP, SimParams};

fn params_dt(dt: f32) -> SimParams {
    SimParams {
        dt,
        inv_dt: 1.0 / dt,
        inv_dt_sq: 1.0 / (dt * dt),
        constraint_batch_count: 0,
        num_particles: 1,
        num_tris: 0,
        jacobi_omega: 1.0,
        inner_iterations: 1,
        thickness: 0.04,
        coll_scale: 0.0,
        _pad_before_gravity: Vec2::ZERO,
        gravity: Vec4::new(0.0, -10.0, 0.0, 0.0),
        grab_target: Vec4::ZERO,
        grab_idx: -1,
        grab_active: 0,
        grab_stiffness: 0.0,
        _pad_legacy_floor: 0.0,
        linear_drag_per_sec: 0.0,
        constraint_batch_idx: 0,
        _uniform_pad_vec2_u: glam::UVec2::ZERO,
        _uniform_pad_vec2_f: Vec2::ZERO,
        _uniform_encase_reserve: glam::UVec2::ZERO,
    }
}

#[test]
fn clamp_delta_vec_caps_long_corrections() {
    let d = clamp_delta_vec(Vec3::new(1.0, 0.0, 0.0));
    assert!((d.length() - CORRECTION_CAP).abs() < 1e-5);
}

#[test]
fn clamp_delta_vec_passes_short_corrections() {
    let v = Vec3::new(0.1, 0.0, 0.0);
    assert_eq!(clamp_delta_vec(v), v);
}

#[test]
fn predict_applies_gravity_and_leaves_pinned_still() {
    let p = params_dt(1.0 / 60.0);
    let (pos, vel) = predict_position(
        &p,
        0,
        Vec4::ZERO,
        Vec4::ZERO,
        1.0,
    );
    assert!(xyz(vel).y < 0.0);
    assert!(xyz(pos).y < 0.0);

    let (pin_pos, pin_vel) = predict_position(&p, 0, Vec4::new(1.0, 2.0, 3.0, 0.0), Vec4::ONE, 0.0);
    assert_eq!(pin_pos, Vec4::new(1.0, 2.0, 3.0, 0.0));
    assert_eq!(pin_vel, Vec4::ONE);
}

#[test]
fn predict_grab_pulls_toward_target() {
    let mut p = params_dt(1.0 / 60.0);
    p.gravity = Vec4::ZERO;
    p.grab_active = 1;
    p.grab_idx = 0;
    p.grab_target = Vec4::new(1.0, 0.0, 0.0, 0.0);
    p.grab_stiffness = 1.0;
    let (pos, _) = predict_position(&p, 0, Vec4::ZERO, Vec4::ZERO, 1.0);
    assert!(xyz(pos).x > 0.0);
}

#[test]
fn grab_particle_is_kinematic_for_constraints() {
    use crate::common::effective_inv_mass;
    let mut p = params_dt(1.0 / 60.0);
    p.grab_active = 1;
    p.grab_idx = 3;
    assert_eq!(effective_inv_mass(&p, 3, 1.0), 0.0);
    assert_eq!(effective_inv_mass(&p, 2, 1.0), 1.0);
}

#[test]
fn xpbd_delta_lambda_zero_compliance_matches_macklin() {
    // len=2, rest=1, w_i=w_j=1, α̃=0 → Δλ = -C/∑w = -0.5
    let dlam = xpbd_distance_delta_lambda(2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    assert!((dlam - (-0.5)).abs() < 1e-5, "Δλ={dlam}");
}

#[test]
fn xpbd_delta_lambda_zero_when_both_pinned() {
    assert_eq!(
        xpbd_distance_delta_lambda(2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        0.0
    );
}

#[test]
fn xpbd_delta_lambda_hard_compress_floor() {
    // len=0.5, rest=1 → below 0.70 floor; compliance ignored → Δλ = 0.10
    let dlam = xpbd_distance_delta_lambda(0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0);
    assert!((dlam - 0.10).abs() < 1e-5, "Δλ={dlam}");
}

#[test]
fn self_collision_non_local_separates_to_thickness_shell() {
    use crate::common::self_collision_separation;
    let thickness = 0.04;
    let rest_i = Vec3::ZERO;
    let rest_j = Vec3::new(0.2, 0.0, 0.0); // non-local
    let p_i = Vec3::ZERO;
    let p_j = Vec3::new(0.01, 0.0, 0.0);
    let (hit, di, dj) =
        self_collision_separation(thickness, 1.0, p_i, p_j, rest_i, rest_j, 1.0, 1.0);
    assert!(hit);
    let p_i2 = p_i + di;
    let p_j2 = p_j + dj;
    let d = (p_j2 - p_i2).length();
    assert!(
        (d - thickness).abs() < 1e-4,
        "expected full thickness separation, got {d}"
    );
}

#[test]
fn collision_hash_packs_cell_and_neighbor_offsets() {
    let grid = CollGridUniform {
        grid_origin_pad: Vec4::new(0.0, 0.0, 0.0, 0.0),
        inv_cell: 10.0,
        num_cells: 8,
        num_particles: 1,
        gx: 2,
        gy: 2,
        gz: 2,
        radix_digits: 1,
        _align_pad: 0,
        _reserved: UVec4::ZERO,
    };
    // Point in cell (1,0,0) → flat index 1
    assert_eq!(
        collision_flat_packed(&grid, Vec3::new(0.15, 0.01, 0.01)),
        1
    );
    assert_eq!(neighbor_flat(&grid, 1, -1, 0, 0), 0);
    assert_eq!(neighbor_flat(&grid, 0, -1, 0, 0), grid.num_cells);
}
