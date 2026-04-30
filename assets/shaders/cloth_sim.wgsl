struct SimParams {
    dt: f32,
    inv_dt: f32,
    inv_dt_sq: f32,
    constraint_batch_count: u32,
    num_particles: u32,
    num_tris: u32,
    jacobi_omega: f32,
    inner_iterations: u32,
    thickness: f32,
    coll_scale: f32,
    /// Aligns **`gravity`** to 16 bytes (matches Rust `ClothSimParamsGpu`).
    _pad_before_gravity: vec2<f32>,
    gravity: vec4<f32>,
    grab_target: vec4<f32>,
    grab_idx: i32,
    grab_active: u32,
    grab_stiffness: f32,
    floor_y: f32,
    linear_drag_per_sec: f32,
    // Unused (legacy layout); real batch index is binding(19) dynamic uniform `gs_dyn_batch`.
    constraint_batch_idx: u32,
    _uniform_pad_vec2_u: vec2<u32>,
    _uniform_pad_vec2_f: vec2<f32>,
    _uniform_encase_reserve: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> sim_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> jac_state: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> prev_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> rest_pos: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(7) var<storage, read> constraint_batch_offsets: array<u32>;
@group(0) @binding(8) var<storage, read> constraint_i: array<u32>;
@group(0) @binding(9) var<storage, read> constraint_j: array<u32>;
@group(0) @binding(10) var<storage, read> constraint_rest: array<f32>;
@group(0) @binding(11) var<storage, read> constraint_comp: array<f32>;
@group(0) @binding(12) var<storage, read_write> constraint_lambda: array<f32>;
@group(0) @binding(13) var<storage, read_write> constraint_delta_lambda: array<f32>;
@group(0) @binding(14) var<storage, read> tri_indices: array<u32>;
@group(0) @binding(15) var<storage, read_write> render_positions: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> render_normals: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read_write> atomic_coll: array<atomic<i32>>;
@group(0) @binding(18) var<storage, read_write> atomic_norm: array<atomic<i32>>;
/// One 256‑byte dynamic slot (`min_uniform_buffer_offset_alignment`). `head.x` holds the batch index.
/// `array<u32>` padding is invalid in uniform address space (stride must be ≥16).
struct GsDynBatchUniform {
    head: vec4<u32>,
    _pad_bulk: array<vec4<u32>, 15>,
}
@group(0) @binding(19) var<uniform> gs_dyn_batch: GsDynBatchUniform;

const FIXSCALE: i32 = 10000;
/// Match `JACOBI_CORRECTION_CAP` in `cloth_compute.rs` (applied per particle per edge correction).
const GS_CORRECTION_CAP: f32 = 0.28;

fn clamp_delta_vec(dx: vec3<f32>) -> vec3<f32> {
    let ml = length(dx);
    if (ml > GS_CORRECTION_CAP && ml > 0.0) {
        return dx * (GS_CORRECTION_CAP / ml);
    }
    return dx;
}

fn xpbd_predict_then_write_jac_row(i: u32) {
    let w = inv_mass[i];
    if (w <= 0.0) {
        prev_pos[i] = sim_pos[i];
        jac_state[i] = sim_pos[i];
        return;
    }
    var v = velocities[i].xyz;
    v += params.gravity.xyz * params.dt;
    let speed = length(v);
    let max_v = 12.0;
    if (speed > max_v) {
        v = v * (max_v / speed);
    }
    prev_pos[i] = sim_pos[i];
    var p = sim_pos[i].xyz + v * params.dt;
    if (p.y < params.floor_y) {
        p.y = params.floor_y;
    }
    if (params.grab_active != 0u && i32(i) == params.grab_idx) {
        let to_t = params.grab_target.xyz - p;
        p = p + to_t * params.grab_stiffness;
    }
    sim_pos[i] = vec4<f32>(p, 0.0);
    velocities[i] = vec4<f32>(v, 0.0);
    jac_state[i] = sim_pos[i];
}

/// Gravity / floor / grab integration, then copy **`sim_pos` → `jac_state`** so GS reads fresh positions in one barrier.
@compute @workgroup_size(64, 1, 1)
fn predict_copy_sim_to_jac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    xpbd_predict_then_write_jac_row(i);
}

@compute @workgroup_size(64, 1, 1)
fn copy_jac_to_sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    sim_pos[i] = jac_state[i];
}

// Zero λ before each GS inner iteration.
@compute @workgroup_size(64, 1, 1)
fn clear_constraint_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&constraint_lambda);
    if (i >= n) {
        return;
    }
    constraint_lambda[i] = 0.0;
    constraint_delta_lambda[i] = 0.0;
}

// XPBD distance constraints for one GS color batch; `gs_dyn_batch.head.x` via dynamic uniform offset.
@compute @workgroup_size(128, 1, 1)
fn gs_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gs_dyn_batch.head.x;
    if (params.constraint_batch_count == 0u || b >= params.constraint_batch_count) {
        return;
    }
    let start = constraint_batch_offsets[b];
    let end = constraint_batch_offsets[b + 1u];
    let e = gid.x + start;
    if (e >= end) {
        return;
    }

    let i = constraint_i[e];
    let j = constraint_j[e];
    let w_i = inv_mass[i];
    let w_j = inv_mass[j];
    if (w_j <= 0.0 && w_i <= 0.0) {
        constraint_delta_lambda[e] = 0.0;
        return;
    }
    let rest = constraint_rest[e];
    let compliance = constraint_comp[e];
    let p_i = jac_state[i].xyz;
    let p_j = jac_state[j].xyz;
    var gv = p_i - p_j;
    let len = length(gv);
    if (len < 1e-8) {
        constraint_delta_lambda[e] = 0.0;
        return;
    }
    gv = gv / len;
    let C = len - rest;
    let alpha_t = compliance * params.inv_dt_sq;
    let wsum = w_i + w_j + alpha_t;
    if (wsum < 1e-8) {
        constraint_delta_lambda[e] = 0.0;
        return;
    }
    let lambda_e = constraint_lambda[e];
    let dlam = (-C - alpha_t * lambda_e) / wsum;
    constraint_delta_lambda[e] = dlam;
    constraint_lambda[e] = lambda_e + dlam;

    let omega = params.jacobi_omega;
    var dx_i = omega * gv * w_i * dlam;
    var dx_j = -(omega * gv * w_j * dlam);
    dx_i = clamp_delta_vec(dx_i);
    dx_j = clamp_delta_vec(dx_j);
    if (w_i > 0.0) {
        jac_state[i] = vec4<f32>(p_i + dx_i, 0.0);
    }
    if (w_j > 0.0) {
        jac_state[j] = vec4<f32>(p_j + dx_j, 0.0);
    }
}

@compute @workgroup_size(64, 1, 1)
fn post_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    if (inv_mass[i] <= 0.0) {
        return;
    }
    var v = (sim_pos[i].xyz - prev_pos[i].xyz) * params.inv_dt;
    let damp = exp(-params.linear_drag_per_sec * params.dt);
    v = v * damp;
    velocities[i] = vec4<f32>(v, 0.0);
}

fn coll_row_start(i: u32, n: u32) -> u32 {
    return i * (n - 1u) - (i * (i - 1u)) / 2u;
}

@compute @workgroup_size(256, 1, 1)
fn clear_atomics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.num_particles * 3u) {
        atomicStore(&atomic_coll[i], 0);
    }
}

@compute @workgroup_size(256, 1, 1)
fn collide_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.coll_scale <= 0.0) {
        return;
    }
    let pair_id = gid.x;
    let n = params.num_particles;
    let total = n * (n - 1u) / 2u;
    if (pair_id >= total) {
        return;
    }
    var lo = 0u;
    var hi = n - 2u;
    while (lo < hi) {
        let mid = (lo + hi + 1u) / 2u;
        if (coll_row_start(mid, n) <= pair_id) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    let i = lo;
    let j = pair_id - coll_row_start(i, n) + i + 1u;
    if (i >= n || j >= n || j <= i) {
        return;
    }
    if (inv_mass[i] <= 0.0 && inv_mass[j] <= 0.0) {
        return;
    }
    let thickness_sq = params.thickness * params.thickness;
    let p_i = sim_pos[i].xyz;
    let p_j = sim_pos[j].xyz;
    var d = p_j - p_i;
    let dist2 = dot(d, d);
    if (dist2 > thickness_sq || dist2 < 1e-18) {
        return;
    }
    let r0 = rest_pos[i].xyz;
    let r1 = rest_pos[j].xyz;
    let rest_d = r1 - r0;
    let rest2 = dot(rest_d, rest_d);
    if (dist2 > rest2) {
        return;
    }
    var min_d = params.thickness;
    if (rest2 < thickness_sq) {
        min_d = sqrt(rest2);
    }
    let dist = sqrt(dist2);
    let corr = (min_d - dist) * 0.5 * params.coll_scale;
    if (corr <= 0.0) {
        return;
    }
    d = (d / dist) * corr;
    let w_i = inv_mass[i];
    let w_j = inv_mass[j];
    let inv_w = 1.0 / max(w_i + w_j, 1e-8);
    let di = -d * w_i * inv_w;
    let dj = d * w_j * inv_w;
    if (w_i > 0.0) {
        atomicAdd(&atomic_coll[i * 3u + 0u], i32(di.x * f32(FIXSCALE)));
        atomicAdd(&atomic_coll[i * 3u + 1u], i32(di.y * f32(FIXSCALE)));
        atomicAdd(&atomic_coll[i * 3u + 2u], i32(di.z * f32(FIXSCALE)));
    }
    if (w_j > 0.0) {
        atomicAdd(&atomic_coll[j * 3u + 0u], i32(dj.x * f32(FIXSCALE)));
        atomicAdd(&atomic_coll[j * 3u + 1u], i32(dj.y * f32(FIXSCALE)));
        atomicAdd(&atomic_coll[j * 3u + 2u], i32(dj.z * f32(FIXSCALE)));
    }
}

@compute @workgroup_size(64, 1, 1)
fn collide_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    if (inv_mass[i] <= 0.0) {
        return;
    }
    var sx = f32(atomicLoad(&atomic_coll[i * 3u + 0u])) / f32(FIXSCALE);
    var sy = f32(atomicLoad(&atomic_coll[i * 3u + 1u])) / f32(FIXSCALE);
    var sz = f32(atomicLoad(&atomic_coll[i * 3u + 2u])) / f32(FIXSCALE);
    let max_d = 0.35;
    sx = clamp(sx, -max_d, max_d);
    sy = clamp(sy, -max_d, max_d);
    sz = clamp(sz, -max_d, max_d);
    let p = sim_pos[i].xyz + vec3<f32>(sx, sy, sz);
    sim_pos[i] = vec4<f32>(p, 0.0);
}

@compute @workgroup_size(256, 1, 1)
fn clear_norm_atomics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.num_particles * 3u) {
        atomicStore(&atomic_norm[i], 0);
    }
}

@compute @workgroup_size(64, 1, 1)
fn accumulate_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.num_tris) {
        return;
    }
    let i0 = tri_indices[t * 3u + 0u];
    let i1 = tri_indices[t * 3u + 1u];
    let i2 = tri_indices[t * 3u + 2u];
    let p0 = sim_pos[i0].xyz;
    let p1 = sim_pos[i1].xyz;
    let p2 = sim_pos[i2].xyz;
    let e0 = p1 - p0;
    let e1 = p2 - p0;
    var c = cross(e0, e1);
    let f = 33333;
    atomicAdd(&atomic_norm[i0 * 3u + 0u], i32(c.x * f32(f)));
    atomicAdd(&atomic_norm[i0 * 3u + 1u], i32(c.y * f32(f)));
    atomicAdd(&atomic_norm[i0 * 3u + 2u], i32(c.z * f32(f)));
    atomicAdd(&atomic_norm[i1 * 3u + 0u], i32(c.x * f32(f)));
    atomicAdd(&atomic_norm[i1 * 3u + 1u], i32(c.y * f32(f)));
    atomicAdd(&atomic_norm[i1 * 3u + 2u], i32(c.z * f32(f)));
    atomicAdd(&atomic_norm[i2 * 3u + 0u], i32(c.x * f32(f)));
    atomicAdd(&atomic_norm[i2 * 3u + 1u], i32(c.y * f32(f)));
    atomicAdd(&atomic_norm[i2 * 3u + 2u], i32(c.z * f32(f)));
}

@compute @workgroup_size(64, 1, 1)
fn finalize_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    let f = 33333.0;
    var n = vec3<f32>(
        f32(atomicLoad(&atomic_norm[i * 3u + 0u])) / f,
        f32(atomicLoad(&atomic_norm[i * 3u + 1u])) / f,
        f32(atomicLoad(&atomic_norm[i * 3u + 2u])) / f,
    );
    let ln = length(n);
    if (ln > 1e-8) {
        n = n / ln;
    } else {
        n = vec3<f32>(0.0, 1.0, 0.0);
    }
    render_normals[i] = vec4<f32>(n, 0.0);
    render_positions[i] = sim_pos[i];
}
