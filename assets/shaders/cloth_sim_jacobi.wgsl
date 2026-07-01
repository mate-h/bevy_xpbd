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
    _pad_before_gravity: vec2<f32>,
    gravity: vec4<f32>,
    grab_target: vec4<f32>,
    grab_idx: i32,
    grab_active: u32,
    grab_stiffness: f32,
    floor_y: f32,
    linear_drag_per_sec: f32,
    constraint_batch_idx: u32,
    _uniform_pad_vec2_u: vec2<u32>,
    _uniform_pad_vec2_f: vec2<f32>,
    _uniform_encase_reserve: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> sim_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> jac_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> jac_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> prev_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> rest_pos: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(8) var<storage, read> neighbor_offsets: array<u32>;
@group(0) @binding(9) var<storage, read> neighbor_packed: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> constraint_i: array<u32>;
@group(0) @binding(11) var<storage, read> constraint_j: array<u32>;
@group(0) @binding(12) var<storage, read> constraint_rest: array<f32>;
@group(0) @binding(13) var<storage, read> constraint_comp: array<f32>;
@group(0) @binding(14) var<storage, read_write> constraint_lambda: array<f32>;
@group(0) @binding(15) var<storage, read_write> constraint_delta_lambda: array<f32>;
@group(0) @binding(16) var<storage, read> tri_indices: array<u32>;
@group(0) @binding(17) var<storage, read_write> render_positions: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read_write> render_normals: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read_write> atomic_coll: array<atomic<i32>>;
@group(0) @binding(20) var<storage, read_write> atomic_norm: array<atomic<i32>>;

struct CollGridUniform {
    grid_origin_pad: vec4<f32>,
    inv_cell: f32,
    num_cells: u32,
    num_particles: u32,
    gx: u32,
    gy: u32,
    gz: u32,
    radix_digits: u32,
    _reserved: vec4<u32>,
}

struct CollRadixPassUniform {
    data: vec4<u32>,
}

@group(0) @binding(21) var<uniform> coll_grid_u: CollGridUniform;
@group(0) @binding(22) var<uniform> coll_radix_pass_u: CollRadixPassUniform;
@group(0) @binding(23) var<storage, read_write> coll_radix_hist: array<atomic<u32>, 256>;
@group(0) @binding(24) var<storage, read_write> coll_radix_head: array<atomic<u32>, 256>;
@group(0) @binding(25) var<storage, read_write> coll_perm_ping: array<u32>;
@group(0) @binding(26) var<storage, read_write> coll_perm_pong: array<u32>;
@group(0) @binding(27) var<storage, read_write> coll_cell_start: array<atomic<u32>>;
@group(0) @binding(28) var<storage, read_write> coll_cell_end_exclusive: array<atomic<u32>>;

const FIXSCALE: i32 = 10000;
const JACOBI_CORRECTION_CAP: f32 = 0.28;
const GRAB_MAX_PULL: f32 = 0.065;

fn clamp_delta_vec(dx: vec3<f32>) -> vec3<f32> {
    let ml = length(dx);
    if (ml > JACOBI_CORRECTION_CAP && ml > 0.0) {
        return dx * (JACOBI_CORRECTION_CAP / ml);
    }
    return dx;
}

fn xpbd_predict_then_write_jac_out_row(i: u32) {
    let w = inv_mass[i];
    if (w <= 0.0) {
        prev_pos[i] = sim_pos[i];
        jac_out[i] = sim_pos[i];
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
        var pull = (params.grab_target.xyz - p) * params.grab_stiffness;
        let pl = length(pull);
        if (pl > GRAB_MAX_PULL && pl > 0.0) {
            pull = pull * (GRAB_MAX_PULL / pl);
        }
        p = p + pull;
    }
    sim_pos[i] = vec4<f32>(p, 0.0);
    velocities[i] = vec4<f32>(v, 0.0);
    jac_out[i] = sim_pos[i];
}

@compute @workgroup_size(64, 1, 1)
fn predict_copy_sim_to_jac(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    xpbd_predict_then_write_jac_out_row(i);
}

@compute @workgroup_size(64, 1, 1)
fn copy_jac_to_sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    sim_pos[i] = jac_in[i];
}

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

@compute @workgroup_size(64, 1, 1)
fn jacobi_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    let nc = arrayLength(&constraint_i);
    if (e >= nc) {
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
    let p_i = jac_in[i].xyz;
    let p_j = jac_in[j].xyz;
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
}

@compute @workgroup_size(64, 1, 1)
fn jacobi_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }
    let w_i = inv_mass[i];
    if (w_i <= 0.0) {
        jac_out[i] = jac_in[i];
        return;
    }
    let p_i = jac_in[i].xyz;
    var acc = vec3<f32>(0.0);
    let start = neighbor_offsets[i];
    let end = neighbor_offsets[i + 1u];
    for (var k = start; k < end; k++) {
        let pack = neighbor_packed[k];
        let j = u32(pack.x);
        let w_j = inv_mass[j];
        if (w_j <= 0.0 && w_i <= 0.0) {
            continue;
        }
        let p_j = jac_in[j].xyz;
        var gv = p_i - p_j;
        let len = length(gv);
        if (len < 1e-8) {
            continue;
        }
        gv = gv / len;
        let eid = u32(pack.w);
        let dlam = constraint_delta_lambda[eid];
        acc = acc + gv * w_i * dlam;
    }
    var delta = params.jacobi_omega * acc;
    delta = clamp_delta_vec(delta);
    jac_out[i] = vec4<f32>(p_i + delta, 0.0);
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

@compute @workgroup_size(256, 1, 1)
fn clear_atomics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.num_particles * 3u) {
        atomicStore(&atomic_coll[i], 0);
    }
}

fn collision_flat_packed(p: vec3<f32>) -> u32 {
    let g = coll_grid_u.grid_origin_pad.xyz;
    let q = (p - g) * coll_grid_u.inv_cell;
    let gx = coll_grid_u.gx;
    let gy = coll_grid_u.gy;
    let gz = coll_grid_u.gz;
    let ix = u32(clamp(i32(floor(q.x)), 0, i32(gx) - 1));
    let iy = u32(clamp(i32(floor(q.y)), 0, i32(gy) - 1));
    let iz = u32(clamp(i32(floor(q.z)), 0, i32(gz) - 1));
    return ix + iy * gx + iz * gx * gy;
}

fn neighbor_flat(cid: u32, dx: i32, dy: i32, dz: i32) -> u32 {
    let gx = coll_grid_u.gx;
    let gy = coll_grid_u.gy;
    let gz = coll_grid_u.gz;
    let iz = cid / (gx * gy);
    let t = cid - iz * gx * gy;
    let iy = t / gx;
    let ix = t - iy * gx;
    let nx = i32(ix) + dx;
    let ny = i32(iy) + dy;
    let nz = i32(iz) + dz;
    if (nx < 0 || ny < 0 || nz < 0) {
        return coll_grid_u.num_cells;
    }
    if (nx >= i32(gx) || ny >= i32(gy) || nz >= i32(gz)) {
        return coll_grid_u.num_cells;
    }
    return u32(nx) + u32(ny) * gx + u32(nz) * gx * gy;
}

fn perm_dst_write(tgt: u32, val: u32) {
    let to_a = (coll_radix_pass_u.data.x & 1u) != 0u;
    if (to_a) {
        coll_perm_ping[tgt] = val;
    } else {
        coll_perm_pong[tgt] = val;
    }
}

fn perm_final_read(i: u32) -> u32 {
    let out_ping = (coll_grid_u.radix_digits & 1u) == 0u;
    return select(coll_perm_ping[i], coll_perm_pong[i], out_ping);
}

@compute @workgroup_size(256, 1, 1)
fn coll_cell_bounds_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= coll_grid_u.num_cells) {
        return;
    }
    atomicStore(&coll_cell_start[i], 0xffffffffu);
    atomicStore(&coll_cell_end_exclusive[i], 0u);
}

@compute @workgroup_size(256, 1, 1)
fn coll_perm_identity_ping(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= coll_grid_u.num_particles) {
        return;
    }
    coll_perm_ping[i] = i;
}

@compute @workgroup_size(256, 1, 1)
fn coll_histogram_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 256u) {
        return;
    }
    atomicStore(&coll_radix_hist[i], 0u);
}

@compute @workgroup_size(256, 1, 1)
fn coll_radix_digit_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_idx = gid.x;
    if (in_idx >= coll_grid_u.num_particles) {
        return;
    }
    let rd_pass = coll_radix_pass_u.data.x;
    let pj = select(coll_perm_ping[in_idx], coll_perm_pong[in_idx], (rd_pass & 1u) != 0u);
    let f = collision_flat_packed(sim_pos[pj].xyz);
    let digit = (f >> (rd_pass * 8u)) & 0xffu;
    atomicAdd(&coll_radix_hist[digit], 1u);
}

@compute @workgroup_size(1, 1, 1)
fn coll_radix_exclusive_bases_heads(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }
    var acc = 0u;
    for (var d = 0u; d < 256u; d++) {
        let c = atomicLoad(&coll_radix_hist[d]);
        atomicStore(&coll_radix_head[d], acc);
        acc = acc + c;
    }
}

@compute @workgroup_size(256, 1, 1)
fn coll_radix_digit_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_idx = gid.x;
    if (in_idx >= coll_grid_u.num_particles) {
        return;
    }
    let rd_pass = coll_radix_pass_u.data.x;
    let pj = select(coll_perm_ping[in_idx], coll_perm_pong[in_idx], (rd_pass & 1u) != 0u);
    let f = collision_flat_packed(sim_pos[pj].xyz);
    let digit = (f >> (rd_pass * 8u)) & 0xffu;
    let tgt = atomicAdd(&coll_radix_head[digit], 1u);
    perm_dst_write(tgt, pj);
}

fn sorted_perm_cell_flat(i: u32) -> u32 {
    let pj = perm_final_read(i);
    return collision_flat_packed(sim_pos[pj].xyz);
}

@compute @workgroup_size(256, 1, 1)
fn coll_sorted_build_cell_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let np = coll_grid_u.num_particles;
    if (i >= np) {
        return;
    }
    let cid_here = sorted_perm_cell_flat(i);
    var run_start = (i == 0u);
    if (!run_start) {
        run_start = sorted_perm_cell_flat(i - 1u) != cid_here;
    }
    var run_end = (i == np - 1u);
    if (!run_end) {
        run_end = sorted_perm_cell_flat(i + 1u) != cid_here;
    }
    if (run_start) {
        atomicStore(&coll_cell_start[cid_here], i);
    }
    if (run_end) {
        atomicStore(&coll_cell_end_exclusive[cid_here], i + 1u);
    }
}

fn narrow_self_collision_pair(i: u32, j: u32) {
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

@compute @workgroup_size(256, 1, 1)
fn collide_grid_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.coll_scale <= 0.0) {
        return;
    }
    let i = gid.x;
    if (i >= params.num_particles) {
        return;
    }

    let my_cell = collision_flat_packed(sim_pos[i].xyz);

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nf = neighbor_flat(my_cell, dx, dy, dz);
                if (nf >= coll_grid_u.num_cells) {
                    continue;
                }
                let s = atomicLoad(&coll_cell_start[nf]);
                let e = atomicLoad(&coll_cell_end_exclusive[nf]);
                if (s == 0xffffffffu) {
                    continue;
                }
                var idx = s;
                loop {
                    if (idx >= e) {
                        break;
                    }
                    let j = perm_final_read(idx);
                    if (j > i) {
                        narrow_self_collision_pair(i, j);
                    }
                    idx = idx + 1u;
                }
            }
        }
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
