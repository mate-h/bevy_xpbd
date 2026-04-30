//! GPU cloth XPBD simulation (compute) + render graph dispatch.
//!
//! Distance constraints follow **XPBD** (Macklin et al., Eq. 17–18): α̃ = α/Δt², Δλ = (−C − α̃λ)/(∑ w + α̃),
//! with **Gauss–Seidel** sequencing: corrections write straight into [`jac_state`], one **color batch** at a time
//! (see [`ClothMeshData::constraint_batch_offsets`](crate::mesh_prep::ClothMeshData)). λ is cleared before each
//! inner iteration. Compliance α comes from [`crate::mesh_prep`].
//!
//! **Gauss–Seidel batches:** Each inner iteration clears λ (`clear_constraint_lambda`), then **`gs_edges`** per color batch. All inner iterations for a **substep** live in **`one`** labeled compute pass (`cloth_pass_distance_gauss_seidel`): ordered dispatches imply storage dependencies between clears and batches; dynamic uniform offsets on `binding(19)` (`GS_BATCH_DYNAMIC_STRIDE`) select each GS batch.
//! One such pass runs **per substep** (**not per inner iteration**).
//!
//! **Dispatch counts:** with **`solve_substeps = S`**, **`solve_inner_iterations = I`**, mesh batch count `B`,
//! **`predict_copy_sim_to_jac`** merges integrate + jac copy (**6** tail dispatches/substep vs 7 historically):
//! base **`S * (6 + I * (B + 1)) + 3`** per frame (`ClothSimNode`).
//! Omit **`clear_atomics`** + radix + **`collide_grid_cells`** + **`collide_apply`** when **`coll_scale ≤ 0`** or when substep **`si`**
//! misses **`collision_every_n_substeps - 1 mod N`** (**`collision_every_n_substeps = 1`** = unchanged).
//!
//! **Stability:** Tune [`SUBSTEPS`] / [`INNER_ITERS`] defaults (`to_sim_config`) or **`ClothSimConfig::solve_*`** overrides; stretch α = 0 with few iterations often looks like edge-length “explosion”.
//! [`ClothSimUniforms::jacobi_omega`] scales each distance correction; raise [`JACOBI_CORRECTION_CAP`] only if edges are long.
//!
//! **Note:** Apple Instruments **Metal GPU** exports (`xctrace export` → `metal_gpu_intervals.xml`) typically show **`(wgpu internal) Pre Pass:Compute Command`** once **per dispatch** inside a pass (`cloth_pass_distance_gauss_seidel` aggregates many such rows). Aggregate time by stacking those rows or by **Replay** totals, not only the few top-level labeled passes.
//!
//! **Note:** See **`docs/CLOTH_SIM_STABILITY.md`** for stability history.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    reflect::Reflect,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
            *,
            ShaderType,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        Render, RenderApp, RenderSystems,
    },
    shader::PipelineCacheError,
};
use std::borrow::Cow;
use std::num::NonZero;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::mesh_prep::ClothMeshData;

pub const CLOTH_SHADER: &str = "shaders/cloth_sim.wgsl";
/// XPBD (Müller et al.): use enough substeps so constraint corrections stay well-behaved vs. `dt`.
/// More substeps shrink substep `dt` → XPBD `α̃ = α/dt²` stays moderate and implicit integration is stabler.
pub const SUBSTEPS: u32 = 36;
/// Inner XPBD constraint iterations per substep (GS + coloring).
pub const INNER_ITERS: u32 = 22;
pub const DT: f32 = 1.0 / 60.0;
pub const THICKNESS: f32 = 0.04;
/// Scales overlap correction in **`collide_grid_cells`** / `collide_apply`; `0` disables self-collision.
pub const DEFAULT_COLL_SCALE: f32 = 0.38;
/// Linear air-drag coefficient \(k\) [1/s]: each substep applies `v *= exp(-k · Δt_sub)` in `post_velocity`
/// (matches `dv/dt ≈ -k v` for light viscous / laminar-like damping). **`0`** disables explicit drag.
///
/// The previous fixed factor-per-substep (~0.987) implied ~38% velocity loss **per frame**, far heavier than
/// typical cloth–air damping and made free motion sag slowly after releases.
pub const DEFAULT_LINEAR_AIR_DRAG_PER_SEC: f32 = 1.25;
/// Max length of one **per-endpoint** distance correction in `gs_edges` (each of i/j). Too small vs edge length → drift/explosion.
pub const JACOBI_CORRECTION_CAP: f32 = 0.28;
/// Hard speed clamp after gravity in `predict` (m/s).
pub const PREDICT_MAX_SPEED: f32 = 12.0;
/// Integer scale for GPU `atomicAdd` in self-collision narrow phase (`cloth_sim.wgsl`); CPU sums `f32` directly.
pub const COLLISION_PAIR_FIXSCALE: i32 = 10_000;
/// Clamp on accumulated self-collision displacement per particle per substep (`collide_apply`).
pub const COLLISION_APPLY_CLAMP: f32 = 0.35;

/// Stride between dynamic uniform slots (`min_uniform_buffer_offset_alignment`, 256 on WebGPU).
pub const GS_BATCH_DYNAMIC_STRIDE: u32 = 256;
/// `gs_edges` workgroup width (Metal benefits from denser grids on large batches; must match WGSL `@workgroup_size`).
pub const GS_EDGE_THREADS: u32 = 128;

/// Debug / tooling: pause GPU sim or advance one frame at a time (see example keyboard handler).
/// Starts **running** (`sim_paused = false`); pause with **`P`** in the cloth example if needed.
/// Extracted to the render world so [`ClothSimNode`] can skip dispatch.
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct ClothSimControl {
    /// When true, the render-graph compute pass is skipped until `step_serial` increases.
    pub sim_paused: bool,
    /// Press "step" (example: N) to increment; each new value runs exactly one sim pass while paused.
    pub step_serial: u64,
}

impl Default for ClothSimControl {
    fn default() -> Self {
        Self {
            sim_paused: false,
            step_serial: 0,
        }
    }
}

/// CPU-side config extracted to the render world.
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct ClothSimConfig {
    /// XPBD substeps per rendered frame (`uniforms.dt` is [`DT`] / `solve_substeps`).
    ///
    /// More steps → stabler stiff constraints but more GPU dispatches (linear in this field).
    pub solve_substeps: u32,
    /// Gauss–Seidel / distance iterations per substep (clears λ, then **`B`** `gs_edges` batches).
    pub solve_inner_iterations: u32,
    /// With **`N = collision_every_n_substeps.max(1)`**, run **`clear_atomics` → (radix sort + `collide_grid_cells`) → `collide_apply`**
    /// only when **`si % N == N - 1`** ( **`N = 1`** = every substep). Skipped entirely when **`coll_scale ≤ 0`** on the extracted uniform.
    pub collision_every_n_substeps: u32,
    pub num_particles: u32,
    pub num_tris: u32,
    pub num_distance_constraints: u32,
    /// Spatial hash bounds for **`collide_grid_cells`**: **[`mesh_prep::derive_collision_grid`]** at load time (**`rest_pos`** + thickness).
    pub coll_grid_origin: Vec3,
    pub coll_grid_inv_cell: f32,
    pub coll_grid_dims: [u32; 3],
    pub coll_num_cells: u32,
    /// Digit passes for radix sort by flat cell (`ceil(bits(flat_max))/8`).
    pub coll_radix_digits: u32,
    pub constraint_batch_offsets: Vec<u32>,
    pub constraint_batch_count: u32,
    pub constraint_i: Vec<u32>,
    pub constraint_j: Vec<u32>,
    pub constraint_rest_len: Vec<f32>,
    pub constraint_compliance: Vec<f32>,
    pub tri_indices: Vec<u32>,
    pub inv_mass: Vec<f32>,
    pub rest_pos: Vec<Vec4>,
    pub initial_pos: Vec<Vec4>,
    pub render_positions: Handle<ShaderStorageBuffer>,
    pub render_normals: Handle<ShaderStorageBuffer>,
}

#[derive(Resource, Clone, ExtractResource, Reflect, ShaderType)]
pub struct ClothSimUniforms {
    pub dt: f32,
    pub inv_dt: f32,
    pub num_particles: u32,
    pub num_tris: u32,
    pub jacobi_omega: f32,
    pub inner_iterations: u32,
    pub thickness: f32,
    /// Pair overlap correction scale; use **`0`** to disable self-collision ([`DEFAULT_COLL_SCALE`] when enabled).
    pub coll_scale: f32,
    /// Acceleration in `predict` (WGSL `.xyz`). Default is ~**9.81 m/s²** downward (−Y).
    pub gravity: Vec4,
    pub grab_target: Vec4,
    pub grab_idx: i32,
    pub grab_active: u32,
    pub grab_stiffness: f32,
    pub floor_y: f32,
    /// Air-drag coefficient \(k\) [s⁻¹]; **`0`** = none (see [`DEFAULT_LINEAR_AIR_DRAG_PER_SEC`]).
    pub linear_drag_per_sec: f32,
    /// Legacy field in main uniform (unused); GS batch lives in **`binding(19)`** dynamic uniform slots.
    pub constraint_batch_idx: u32,
}

impl Default for ClothSimUniforms {
    fn default() -> Self {
        let sdt = DT / SUBSTEPS as f32;
        Self {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            num_particles: 0,
            num_tris: 0,
            jacobi_omega: 1.0,
            inner_iterations: INNER_ITERS,
            thickness: THICKNESS,
            coll_scale: DEFAULT_COLL_SCALE,
            gravity: Vec4::new(0.0, -9.81, 0.0, 0.0),
            grab_target: Vec4::ZERO,
            grab_idx: -1,
            grab_active: 0,
            grab_stiffness: 0.45,
            floor_y: -2.0,
            linear_drag_per_sec: DEFAULT_LINEAR_AIR_DRAG_PER_SEC,
            constraint_batch_idx: 0,
        }
    }
}

/// WGSL `SimParams`. Must match WGSL packing in `cloth_sim.wgsl`; written with [`bytemuck`] (not encase).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothSimParamsGpu {
    pub dt: f32,
    pub inv_dt: f32,
    pub inv_dt_sq: f32,
    /// Equal to `constraint_offsets.len().saturating_sub(1)`; batch index `b` valid when `b < constraint_batch_count`.
    pub constraint_batch_count: u32,
    pub num_particles: u32,
    pub num_tris: u32,
    pub jacobi_omega: f32,
    pub inner_iterations: u32,
    pub thickness: f32,
    pub coll_scale: f32,
    /// Padding before `gravity` so **`[f32; 4]`** is 16-byte aligned (`Pod` forbids implicit padding).
    pub _pad_before_gravity: [f32; 2],
    pub gravity: [f32; 4],
    pub grab_target: [f32; 4],
    pub grab_idx: i32,
    pub grab_active: u32,
    pub grab_stiffness: f32,
    pub floor_y: f32,
    pub linear_drag_per_sec: f32,
    pub constraint_batch_idx: u32,
    pub _uniform_pad_vec2_u: [u32; 2],
    pub _uniform_pad_vec2_f: [f32; 2],
    pub _uniform_encase_reserve: [u32; 2],
}

/// **`coll_grid_u`** uniform (`cloth_sim.wgsl`) — spatial hash bookkeeping for **`collide_grid_cells`**.
///
/// WGSL aligns the trailing **`vec4<u32>`_reserved** — keep **`_align_pad`** + **`_reserved`** in sync with `cloth_sim.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothCollGridGpu {
    pub grid_origin_pad: [f32; 4],
    pub inv_cell: f32,
    pub num_cells: u32,
    pub num_particles: u32,
    pub gx: u32,
    pub gy: u32,
    pub gz: u32,
    pub radix_digits: u32,
    pub _align_pad: [u8; 4],
    pub _reserved: [u32; 4],
}

impl ClothCollGridGpu {
    pub fn pack_from_config(cfg: &ClothSimConfig) -> Self {
        let o = cfg.coll_grid_origin;
        Self {
            grid_origin_pad: [o.x, o.y, o.z, 0.0],
            inv_cell: cfg.coll_grid_inv_cell,
            num_cells: cfg.coll_num_cells,
            num_particles: cfg.num_particles,
            gx: cfg.coll_grid_dims[0],
            gy: cfg.coll_grid_dims[1],
            gz: cfg.coll_grid_dims[2],
            radix_digits: cfg.coll_radix_digits,
            _align_pad: [0u8; 4],
            _reserved: [0u32; 4],
        }
    }
}

/// WGSL **`CollRadixPassUniform`**: radix pass index lives in **`data.x`** (`vec4<u32>` for uniform alignment).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothCollRadixPassGpu {
    pub data: [u32; 4],
}

impl ClothSimParamsGpu {
    /// Fills GPU uniform bytes; **`constraint_batch_count`** must match **`constraint_batch_offsets`** (length − 1).
    pub fn pack(uniforms: &ClothSimUniforms, constraint_batch_count: u32) -> Self {
        let inv_dt = uniforms.inv_dt;
        Self {
            dt: uniforms.dt,
            inv_dt,
            inv_dt_sq: inv_dt * inv_dt,
            constraint_batch_count,
            num_particles: uniforms.num_particles,
            num_tris: uniforms.num_tris,
            jacobi_omega: uniforms.jacobi_omega,
            inner_iterations: uniforms.inner_iterations,
            thickness: uniforms.thickness,
            coll_scale: uniforms.coll_scale,
            _pad_before_gravity: [0.0, 0.0],
            gravity: uniforms.gravity.to_array(),
            grab_target: uniforms.grab_target.to_array(),
            grab_idx: uniforms.grab_idx,
            grab_active: uniforms.grab_active,
            grab_stiffness: uniforms.grab_stiffness,
            floor_y: uniforms.floor_y,
            linear_drag_per_sec: uniforms.linear_drag_per_sec,
            constraint_batch_idx: uniforms.constraint_batch_idx,
            _uniform_pad_vec2_u: [0, 0],
            _uniform_pad_vec2_f: [0.0, 0.0],
            _uniform_encase_reserve: [0, 0],
        }
    }
}

#[derive(Resource)]
pub struct ClothSimBuffers {
    /// `SimParams` (`var<uniform>`) — bytes from [`ClothSimParamsGpu`].
    pub params_uniform: Buffer,
    pub sim_pos: Buffer,
    pub jac_state: Buffer,
    pub prev: Buffer,
    pub vel: Buffer,
    pub rest: Buffer,
    pub inv_mass: Buffer,
    pub constraint_batch_offsets: Buffer,
    pub constraint_i: Buffer,
    pub constraint_j: Buffer,
    pub constraint_rest: Buffer,
    pub constraint_comp: Buffer,
    pub constraint_lambda: Buffer,
    pub constraint_delta_lambda: Buffer,
    pub tri: Buffer,
    pub atomic_coll: Buffer,
    pub atomic_norm: Buffer,
    /// `binding(19)` lut: **`batch_count`** slots × [`GS_BATCH_DYNAMIC_STRIDE`] bytes (batch indices 0…).
    pub gs_batch_dyn: Buffer,
    pub coll_grid_uniform: Buffer,
    pub coll_radix_pass_uniform: Buffer,
    pub coll_radix_hist: Buffer,
    pub coll_radix_head: Buffer,
    pub coll_perm_ping: Buffer,
    pub coll_perm_pong: Buffer,
    pub coll_cell_start: Buffer,
    pub coll_cell_end_exclusive: Buffer,
}

#[derive(Resource)]
struct ClothPipeline {
    layout: BindGroupLayoutDescriptor,
    predict_copy_sim_to_jac: CachedComputePipelineId,
    copy_jac_to_sim: CachedComputePipelineId,
    clear_constraint_lambda: CachedComputePipelineId,
    gs_edges: CachedComputePipelineId,
    post_velocity: CachedComputePipelineId,
    clear_atomics: CachedComputePipelineId,
    coll_cell_bounds_clear: CachedComputePipelineId,
    coll_perm_identity_ping: CachedComputePipelineId,
    coll_histogram_clear: CachedComputePipelineId,
    coll_radix_digit_count: CachedComputePipelineId,
    coll_radix_exclusive_bases_heads: CachedComputePipelineId,
    coll_radix_digit_scatter: CachedComputePipelineId,
    coll_sorted_build_cell_ranges: CachedComputePipelineId,
    collide_grid_cells: CachedComputePipelineId,
    collide_apply: CachedComputePipelineId,
    clear_norm_atomics: CachedComputePipelineId,
    accumulate_normals: CachedComputePipelineId,
    finalize_normals: CachedComputePipelineId,
}

#[derive(Resource)]
struct ClothBindGroups {
    cloth: BindGroup,
}

#[derive(Resource, Default, Clone, Copy)]
enum ClothLoadState {
    #[default]
    Loading,
    Ready,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ClothSimLabel;

fn sync_cloth_solve_budget_to_uniforms(
    cfg: Option<Res<ClothSimConfig>>,
    mut uniforms: ResMut<ClothSimUniforms>,
) {
    let Some(cfg) = cfg else {
        return;
    };
    let s = cfg.solve_substeps.max(1);
    let i = cfg.solve_inner_iterations.max(1);
    uniforms.dt = DT / s as f32;
    uniforms.inv_dt = 1.0 / uniforms.dt;
    uniforms.inner_iterations = i;
}

pub struct ClothComputePlugin;

impl Plugin for ClothComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClothSimControl>();
        app.add_systems(
            PreUpdate,
            sync_cloth_solve_budget_to_uniforms.run_if(resource_exists::<ClothSimConfig>),
        );
        app.add_plugins((
            ExtractResourcePlugin::<ClothSimConfig>::default(),
            ExtractResourcePlugin::<ClothSimUniforms>::default(),
            ExtractResourcePlugin::<ClothSimControl>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ClothLoadState>();
        render_app
            .add_systems(
                Render,
                (
                    (init_cloth_sim, check_cloth_pipeline)
                        .chain()
                        .in_set(RenderSystems::Prepare),
                    prepare_cloth_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ClothSimLabel, ClothSimNode::default());
        render_graph.add_node_edge(ClothSimLabel, bevy::render::graph::CameraDriverLabel);
    }
}

fn init_cloth_sim(
    mut commands: Commands,
    config: Option<Res<ClothSimConfig>>,
    uniforms: Res<ClothSimUniforms>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    existing: Option<Res<ClothSimBuffers>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(config) = config else {
        return;
    };
    if config.num_particles == 0 {
        return;
    }

    let n = config.num_particles as usize;
    let n3 = n * 3;
    let vec4_sz = |count: usize| (count * 16) as u64;
    let f32_sz = |count: usize| (count * 4) as u64;
    let u32_sz = |count: usize| (count * 4) as u64;

    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let params_sz = std::mem::size_of::<ClothSimParamsGpu>() as u64;
    let params_uniform = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sim_params_uniform"),
        size: params_sz,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let initial_params =
        ClothSimParamsGpu::pack(uniforms.as_ref(), config.constraint_batch_count);
    render_queue.write_buffer(
        &params_uniform,
        0,
        bytemuck::bytes_of(&initial_params),
    );

    let sim_pos = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sim_pos"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let jac_state = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_jac_state"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let prev = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_prev"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let vel = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_vel"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let rest = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_rest"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let inv_mass_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_inv_mass"),
        size: f32_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let batch_offs_len = config.constraint_batch_offsets.len().max(2);
    let constraint_batch_offsets_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_batch_offsets"),
        size: u32_sz(batch_offs_len),
        usage,
        mapped_at_creation: false,
    });
    let ec = config.num_distance_constraints as usize;
    let ec_store = ec.max(1);
    let constraint_i_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_i"),
        size: u32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let constraint_j_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_j"),
        size: u32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let constraint_rest_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_rest"),
        size: f32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let constraint_comp_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_comp"),
        size: f32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let constraint_lambda_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_lambda"),
        size: f32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let constraint_delta_lambda_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_constraint_delta_lambda"),
        size: f32_sz(ec_store),
        usage,
        mapped_at_creation: false,
    });
    let tri = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_tri"),
        size: u32_sz(config.tri_indices.len()),
        usage,
        mapped_at_creation: false,
    });
    let atomic_coll = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_atomic_coll"),
        size: (n3 * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let atomic_norm = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_atomic_norm"),
        size: (n3 * 4) as u64,
        usage,
        mapped_at_creation: false,
    });

    let nb_lut = config.constraint_batch_count.max(1) as usize;
    let gs_dyn_bytes = (GS_BATCH_DYNAMIC_STRIDE as usize).saturating_mul(nb_lut).max(GS_BATCH_DYNAMIC_STRIDE as usize);
    let mut gs_dyn_lut = vec![0u8; gs_dyn_bytes];
    for bat in 0..(config.constraint_batch_count as usize) {
        let o = bat * GS_BATCH_DYNAMIC_STRIDE as usize;
        gs_dyn_lut[o..o + 4].copy_from_slice(&(bat as u32).to_le_bytes());
    }
    let gs_batch_dyn = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_gs_batch_dyn"),
        size: gs_dyn_bytes as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&gs_batch_dyn, 0, &gs_dyn_lut);

    let coll_grid_u_sz = std::mem::size_of::<ClothCollGridGpu>() as u64;
    let coll_radix_pass_sz = std::mem::size_of::<ClothCollRadixPassGpu>() as u64;
    let radix_arr = 256u64 * 4;
    let radix_nz = NonZero::new(radix_arr).expect("radix atomic buffer size");
    let nc = config.coll_num_cells.max(1) as usize;
    let perm_nz = NonZero::new(u32_sz(n)).expect("perm buffer");
    let cell_nz = NonZero::new(u32_sz(nc)).expect("cell buffer");

    let coll_grid_uniform = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_grid_uniform"),
        size: coll_grid_u_sz,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let grid_gpu = ClothCollGridGpu::pack_from_config(&*config);
    render_queue.write_buffer(&coll_grid_uniform, 0, bytemuck::bytes_of(&grid_gpu));

    let coll_radix_pass_uniform = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_radix_pass_uniform"),
        size: coll_radix_pass_sz,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(
        &coll_radix_pass_uniform,
        0,
        bytemuck::bytes_of(&ClothCollRadixPassGpu { data: [0u32; 4] }),
    );

    let coll_radix_hist = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_radix_hist"),
        size: radix_arr,
        usage,
        mapped_at_creation: false,
    });
    let coll_radix_head = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_radix_head"),
        size: radix_arr,
        usage,
        mapped_at_creation: false,
    });
    let coll_perm_ping = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_perm_ping"),
        size: u32_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let coll_perm_pong = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_perm_pong"),
        size: u32_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let coll_cell_start = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_cell_start"),
        size: u32_sz(nc),
        usage,
        mapped_at_creation: false,
    });
    let coll_cell_end_exclusive = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_cell_end_exclusive"),
        size: u32_sz(nc),
        usage,
        mapped_at_creation: false,
    });

    if ec > 0 {
        render_queue.write_buffer(
            &constraint_i_buf,
            0,
            bytemuck::cast_slice::<u32, u8>(&config.constraint_i),
        );
        render_queue.write_buffer(
            &constraint_j_buf,
            0,
            bytemuck::cast_slice::<u32, u8>(&config.constraint_j),
        );
        render_queue.write_buffer(
            &constraint_rest_buf,
            0,
            bytemuck::cast_slice::<f32, u8>(&config.constraint_rest_len),
        );
        render_queue.write_buffer(
            &constraint_comp_buf,
            0,
            bytemuck::cast_slice::<f32, u8>(&config.constraint_compliance),
        );
    }
    render_queue.write_buffer(
        &constraint_lambda_buf,
        0,
        &vec![0u8; f32_sz(ec_store) as usize],
    );
    render_queue.write_buffer(
        &constraint_delta_lambda_buf,
        0,
        &vec![0u8; f32_sz(ec_store) as usize],
    );
    let mut batch_offs_upload = config.constraint_batch_offsets.clone();
    batch_offs_upload.resize(batch_offs_len, 0);
    render_queue.write_buffer(
        &constraint_batch_offsets_buf,
        0,
        bytemuck::cast_slice::<u32, u8>(&batch_offs_upload),
    );
    render_queue.write_buffer(
        &tri,
        0,
        bytemuck::cast_slice::<u32, u8>(&config.tri_indices),
    );
    render_queue.write_buffer(
        &inv_mass_buf,
        0,
        bytemuck::cast_slice::<f32, u8>(&config.inv_mass),
    );
    render_queue.write_buffer(
        &rest,
        0,
        bytemuck::cast_slice::<Vec4, u8>(&config.rest_pos),
    );
    let ip = bytemuck::cast_slice::<Vec4, u8>(&config.initial_pos);
    render_queue.write_buffer(&sim_pos, 0, ip);
    render_queue.write_buffer(&jac_state, 0, ip);
    render_queue.write_buffer(&prev, 0, ip);
    render_queue.write_buffer(&vel, 0, &vec![0u8; vec4_sz(n) as usize]);

    let layout = BindGroupLayoutDescriptor::new(
        "cloth_sim",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(
                    false,
                    NonZero::new(std::mem::size_of::<ClothSimParamsGpu>() as u64),
                ),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                uniform_buffer_sized(true, NonZero::new(GS_BATCH_DYNAMIC_STRIDE as u64)),
                uniform_buffer_sized(
                    false,
                    NonZero::new(std::mem::size_of::<ClothCollGridGpu>() as u64),
                ),
                uniform_buffer_sized(
                    false,
                    NonZero::new(std::mem::size_of::<ClothCollRadixPassGpu>() as u64),
                ),
                storage_buffer_sized(false, Some(radix_nz)),
                storage_buffer_sized(false, Some(radix_nz)),
                storage_buffer_sized(false, Some(perm_nz)),
                storage_buffer_sized(false, Some(perm_nz)),
                storage_buffer_sized(false, Some(cell_nz)),
                storage_buffer_sized(false, Some(cell_nz)),
            ),
        ),
    );

    let shader = asset_server.load(CLOTH_SHADER);

    macro_rules! cp {
        ($label:literal, $name:literal) => {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Borrowed($label)),
                layout: vec![layout.clone()],
                shader: shader.clone(),
                entry_point: Some(Cow::Borrowed($name)),
                ..default()
            })
        };
    }

    commands.insert_resource(ClothPipeline {
        layout: layout.clone(),
        predict_copy_sim_to_jac: cp!(
            "cloth_cs_predict_copy_sim_to_jac",
            "predict_copy_sim_to_jac"
        ),
        copy_jac_to_sim: cp!("cloth_cs_copy_jac_to_sim", "copy_jac_to_sim"),
        clear_constraint_lambda: cp!(
            "cloth_cs_clear_constraint_lambda",
            "clear_constraint_lambda"
        ),
        gs_edges: cp!("cloth_cs_gs_edges", "gs_edges"),
        post_velocity: cp!("cloth_cs_post_velocity", "post_velocity"),
        clear_atomics: cp!("cloth_cs_clear_atomics", "clear_atomics"),
        coll_cell_bounds_clear: cp!(
            "cloth_cs_coll_cell_bounds_clear",
            "coll_cell_bounds_clear"
        ),
        coll_perm_identity_ping: cp!(
            "cloth_cs_coll_perm_identity_ping",
            "coll_perm_identity_ping"
        ),
        coll_histogram_clear: cp!("cloth_cs_coll_histogram_clear", "coll_histogram_clear"),
        coll_radix_digit_count: cp!(
            "cloth_cs_coll_radix_digit_count",
            "coll_radix_digit_count"
        ),
        coll_radix_exclusive_bases_heads: cp!(
            "cloth_cs_coll_radix_exclusive_bases_heads",
            "coll_radix_exclusive_bases_heads"
        ),
        coll_radix_digit_scatter: cp!(
            "cloth_cs_coll_radix_digit_scatter",
            "coll_radix_digit_scatter"
        ),
        coll_sorted_build_cell_ranges: cp!(
            "cloth_cs_coll_sorted_build_cell_ranges",
            "coll_sorted_build_cell_ranges"
        ),
        collide_grid_cells: cp!("cloth_cs_collide_grid_cells", "collide_grid_cells"),
        collide_apply: cp!("cloth_cs_collide_apply", "collide_apply"),
        clear_norm_atomics: cp!("cloth_cs_clear_norm_atomics", "clear_norm_atomics"),
        accumulate_normals: cp!("cloth_cs_accumulate_normals", "accumulate_normals"),
        finalize_normals: cp!("cloth_cs_finalize_normals", "finalize_normals"),
    });

    commands.insert_resource(ClothSimBuffers {
        params_uniform,
        sim_pos,
        jac_state,
        prev,
        vel,
        rest,
        inv_mass: inv_mass_buf,
        constraint_batch_offsets: constraint_batch_offsets_buf,
        constraint_i: constraint_i_buf,
        constraint_j: constraint_j_buf,
        constraint_rest: constraint_rest_buf,
        constraint_comp: constraint_comp_buf,
        constraint_lambda: constraint_lambda_buf,
        constraint_delta_lambda: constraint_delta_lambda_buf,
        tri,
        atomic_coll,
        atomic_norm,
        gs_batch_dyn,
        coll_grid_uniform,
        coll_radix_pass_uniform,
        coll_radix_hist,
        coll_radix_head,
        coll_perm_ping,
        coll_perm_pong,
        coll_cell_start,
        coll_cell_end_exclusive,
    });
}

fn make_bind_group(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    buffers: &ClothSimBuffers,
    gpu_rp: &GpuShaderStorageBuffer,
    gpu_rn: &GpuShaderStorageBuffer,
) -> BindGroup {
    let gs_dyn_slot =
        BufferSize::new(GS_BATCH_DYNAMIC_STRIDE as u64).expect("stride must fit BufferSize");
    render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            buffers.params_uniform.as_entire_buffer_binding(),
            buffers.sim_pos.as_entire_binding(),
            buffers.jac_state.as_entire_binding(),
            buffers.prev.as_entire_binding(),
            buffers.vel.as_entire_binding(),
            buffers.rest.as_entire_binding(),
            buffers.inv_mass.as_entire_binding(),
            buffers.constraint_batch_offsets.as_entire_binding(),
            buffers.constraint_i.as_entire_binding(),
            buffers.constraint_j.as_entire_binding(),
            buffers.constraint_rest.as_entire_binding(),
            buffers.constraint_comp.as_entire_binding(),
            buffers.constraint_lambda.as_entire_binding(),
            buffers.constraint_delta_lambda.as_entire_binding(),
            buffers.tri.as_entire_binding(),
            gpu_rp.buffer.as_entire_binding(),
            gpu_rn.buffer.as_entire_binding(),
            buffers.atomic_coll.as_entire_binding(),
            buffers.atomic_norm.as_entire_binding(),
            BufferBinding {
                buffer: buffers.gs_batch_dyn.deref(),
                offset: 0,
                size: Some(gs_dyn_slot),
            },
            buffers.coll_grid_uniform.as_entire_buffer_binding(),
            buffers.coll_radix_pass_uniform.as_entire_buffer_binding(),
            buffers.coll_radix_hist.as_entire_binding(),
            buffers.coll_radix_head.as_entire_binding(),
            buffers.coll_perm_ping.as_entire_binding(),
            buffers.coll_perm_pong.as_entire_binding(),
            buffers.coll_cell_start.as_entire_binding(),
            buffers.coll_cell_end_exclusive.as_entire_binding(),
        )),
    )
}

fn prepare_cloth_bind_groups(
    mut commands: Commands,
    pipeline: Option<Res<ClothPipeline>>,
    buffers: Option<Res<ClothSimBuffers>>,
    uniforms: Res<ClothSimUniforms>,
    config: Option<Res<ClothSimConfig>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    gpu_sb: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let Some(buffers) = buffers else {
        return;
    };
    let Some(config) = config.as_ref() else {
        return;
    };
    let Some(gpu_rp) = gpu_sb.get(config.render_positions.id()) else {
        return;
    };
    let Some(gpu_rn) = gpu_sb.get(config.render_normals.id()) else {
        return;
    };

    if config.num_particles == 0 {
        return;
    }

    let gpu_params =
        ClothSimParamsGpu::pack(uniforms.as_ref(), config.constraint_batch_count);
    render_queue.write_buffer(
        &buffers.params_uniform,
        0,
        bytemuck::bytes_of(&gpu_params),
    );

    let gpu_grid = ClothCollGridGpu::pack_from_config(config);
    render_queue.write_buffer(
        &buffers.coll_grid_uniform,
        0,
        bytemuck::bytes_of(&gpu_grid),
    );

    let cloth_bg = make_bind_group(
        &render_device,
        &pipeline_cache,
        &pipeline.layout,
        &buffers,
        gpu_rp,
        gpu_rn,
    );

    commands.insert_resource(ClothBindGroups { cloth: cloth_bg });
}

fn check_cloth_pipeline(
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<ClothPipeline>>,
    mut state: ResMut<ClothLoadState>,
) {
    let Some(pipeline) = pipeline.as_ref() else {
        return;
    };
    if matches!(*state, ClothLoadState::Ready) {
        return;
    }
    for id in [
        pipeline.predict_copy_sim_to_jac,
        pipeline.gs_edges,
        pipeline.collide_grid_cells,
        pipeline.finalize_normals,
    ] {
        match pipeline_cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Ok(_) => {}
            CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => return,
            CachedPipelineState::Err(e) => panic!("cloth shader: {e}"),
            _ => return,
        }
    }
    *state = ClothLoadState::Ready;
}

struct ClothSimNode {
    /// Last `ClothSimControl::step_serial` we dispatched while stepping (or after any run).
    last_ack_step_serial: AtomicU64,
}

impl Default for ClothSimNode {
    fn default() -> Self {
        Self {
            last_ack_step_serial: AtomicU64::new(0),
        }
    }
}

impl render_graph::Node for ClothSimNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let ctrl = world.resource::<ClothSimControl>();
        if ctrl.sim_paused
            && ctrl.step_serial == self.last_ack_step_serial.load(Ordering::Relaxed)
        {
            return Ok(());
        }
        if !matches!(*world.resource::<ClothLoadState>(), ClothLoadState::Ready) {
            return Ok(());
        }
        let Some(bg) = world.get_resource::<ClothBindGroups>() else {
            return Ok(());
        };

        let pipeline = world.resource::<ClothPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let config = world.resource::<ClothSimConfig>();
        let uniforms = world.resource::<ClothSimUniforms>();

        let n = config.num_particles;
        let wg64 = (n + 63) / 64;
        let wg256 = (n * 3 + 255) / 256;
        let wg_tris = (config.num_tris + 63) / 64;
        let wg_n256 = (n + 255) / 256;
        let num_cells = config.coll_num_cells.max(1);
        let wg_cell_clear = (num_cells + 255) / 256;
        let radix_digits = config.coll_radix_digits.max(1);
        let n_constraints = config.num_distance_constraints;
        let wg_constraints = (n_constraints + 63) / 64;
        let num_batches = config.constraint_batch_count as usize;
        let substeps = config.solve_substeps.max(1) as usize;
        let inner_iters = config.solve_inner_iterations.max(1) as usize;
        let coll_tail_enabled = uniforms.coll_scale > 1e-12;
        let col_stride = config.collision_every_n_substeps.max(1) as usize;

        const DYN_IDLE: &[u32] = &[0];
        let encoder = render_context.command_encoder();
        let render_queue = world.resource::<RenderQueue>();
        let collision_buffers = world.resource::<ClothSimBuffers>();

        for si in 0..substeps {
            let run_collision_trio =
                coll_tail_enabled && (si % col_stride == col_stride.saturating_sub(1));

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_integrate_jac_seed"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.predict_copy_sim_to_jac)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg64, 1, 1);
            }

            if n_constraints > 0 && config.constraint_batch_count > 0 {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_distance_gauss_seidel"),
                    timestamp_writes: None,
                });
                for _inner_i in 0..inner_iters {
                    pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.clear_constraint_lambda)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg_constraints, 1, 1);

                    pass.set_pipeline(pipeline_cache.get_compute_pipeline(pipeline.gs_edges).unwrap());
                    for bat in 0..num_batches {
                        let start = config.constraint_batch_offsets[bat] as usize;
                        let end = config.constraint_batch_offsets[bat + 1] as usize;
                        let span = end.saturating_sub(start);
                        if span == 0 {
                            continue;
                        }
                        let t = GS_EDGE_THREADS.max(1) as usize;
                        let wg_batch = ((span + (t - 1)) / t) as u32;
                        let dyn_off = (bat as u32).saturating_mul(GS_BATCH_DYNAMIC_STRIDE);
                        pass.set_bind_group(0, &bg.cloth, &[dyn_off]);
                        pass.dispatch_workgroups(wg_batch, 1, 1);
                    }
                }
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_collision_velocity"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.copy_jac_to_sim)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg64, 1, 1);

                if run_collision_trio {
                    pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.clear_atomics)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg256, 1, 1);

                    pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.coll_perm_identity_ping)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg_n256.max(1), 1, 1);
                }
            }

            if run_collision_trio {
                let cb = collision_buffers;
                for d in 0..radix_digits {
                    let radix_u = ClothCollRadixPassGpu {
                        data: [d, 0, 0, 0],
                    };
                    render_queue.write_buffer(
                        &cb.coll_radix_pass_uniform,
                        0,
                        bytemuck::bytes_of(&radix_u),
                    );

                    let mut rpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cloth_pass_collision_radix"),
                        timestamp_writes: None,
                    });
                    rpass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                    rpass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.coll_histogram_clear)
                            .unwrap(),
                    );
                    rpass.dispatch_workgroups(1, 1, 1);

                    rpass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.coll_radix_digit_count)
                            .unwrap(),
                    );
                    rpass.dispatch_workgroups(wg_n256.max(1), 1, 1);

                    rpass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.coll_radix_exclusive_bases_heads)
                            .unwrap(),
                    );
                    rpass.dispatch_workgroups(1, 1, 1);

                    rpass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.coll_radix_digit_scatter)
                            .unwrap(),
                    );
                    rpass.dispatch_workgroups(wg_n256.max(1), 1, 1);
                }

                let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_collision_grid_narrow"),
                    timestamp_writes: None,
                });
                cpass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                cpass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.coll_cell_bounds_clear)
                        .unwrap(),
                );
                cpass.dispatch_workgroups(wg_cell_clear.max(1), 1, 1);

                cpass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.coll_sorted_build_cell_ranges)
                        .unwrap(),
                );
                cpass.dispatch_workgroups(wg_n256.max(1), 1, 1);

                cpass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.collide_grid_cells)
                        .unwrap(),
                );
                        cpass.dispatch_workgroups(wg_n256.max(1), 1, 1);

                cpass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.collide_apply)
                        .unwrap(),
                );
                cpass.dispatch_workgroups(wg64, 1, 1);
            }

            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_post_velocity"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.post_velocity)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg64, 1, 1);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth_pass_mesh_normals"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.clear_norm_atomics)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg256, 1, 1);

            pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.accumulate_normals)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg_tris, 1, 1);

            pass.set_bind_group(0, &bg.cloth, DYN_IDLE);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.finalize_normals)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);
        }

        self.last_ack_step_serial
            .store(ctrl.step_serial, Ordering::Relaxed);

        Ok(())
    }
}

#[cfg(test)]
mod simulation_data_tests {
    use crate::mesh_prep::grid_cloth_hanging;

    #[test]
    fn cloth_neighbor_and_tri_indices_in_range() {
        let cloth = grid_cloth_hanging(12, 10, 0.05);
        let n = cloth.num_particles;
        for &j in &cloth.neighbor_other {
            assert!(
                j < n,
                "neighbor index {} out of range (num_particles={})",
                j,
                n
            );
        }
        for &t in &cloth.indices {
            assert!(
                t < n,
                "triangle vertex index {} out of range (num_particles={})",
                t,
                n
            );
        }
    }

    /// `neighbor_offsets` must index contiguous slices of real edges — ordering bugs here collapsed the mesh / exploded Jacobi.
    #[test]
    fn cloth_neighbor_slices_match_constraints() {
        use std::collections::HashSet;
        let cloth = grid_cloth_hanging(10, 8, 0.05);
        let n = cloth.num_particles as usize;
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for k in 0..cloth.constraint_i.len() {
            let i = cloth.constraint_i[k];
            let j = cloth.constraint_j[k];
            let a = i.min(j);
            let b = i.max(j);
            edges.insert((a, b));
        }
        for i in 0..n {
            let s = cloth.neighbor_offsets[i] as usize;
            let e = cloth.neighbor_offsets[i + 1] as usize;
            for k in s..e {
                let j = cloth.neighbor_other[k];
                let a = (i as u32).min(j);
                let b = (i as u32).max(j);
                assert!(
                    edges.contains(&(a, b)),
                    "particle {i} neighbor slice lists j={j} but no constraint (a,b)=({a},{b})",
                );
            }
        }
    }
}

#[cfg(test)]
mod uniform_layout_tests {
    use super::{ClothSimParamsGpu, ClothSimUniforms};
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn cloth_sim_uniforms_uniform_buffer_compatible() {
        ClothSimUniforms::assert_uniform_compat();
        let enc = ClothSimUniforms::min_size().get() as usize;
        let gpu = std::mem::size_of::<ClothSimParamsGpu>();
        assert!(
            gpu >= enc,
            "GPU uniform struct ({gpu} bytes) must be at least as large as encase ClothSimUniforms ({enc} bytes)",
        );
    }
}

impl ClothMeshData {
    pub fn to_sim_config(
        &self,
        buf_assets: &mut Assets<ShaderStorageBuffer>,
    ) -> ClothSimConfig {
        let initial_pos: Vec<Vec4> = self
            .positions
            .iter()
            .map(|p| Vec4::new(p.x, p.y, p.z, 0.0))
            .collect();
        let rest_pos: Vec<Vec4> = self
            .rest_positions
            .iter()
            .map(|p| Vec4::new(p.x, p.y, p.z, 0.0))
            .collect();

        let (
            coll_grid_origin,
            coll_grid_inv_cell,
            coll_grid_dims,
            coll_num_cells,
            coll_radix_digits,
        ) = crate::mesh_prep::derive_collision_grid(&rest_pos, THICKNESS);

        let initial_nrm: Vec<Vec4> = self
            .normals
            .iter()
            .map(|v| Vec4::new(v.x, v.y, v.z, 0.0))
            .collect();

        let mut rp = ShaderStorageBuffer::new(
            bytemuck::cast_slice(&initial_pos),
            RenderAssetUsages::RENDER_WORLD,
        );
        rp.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let mut rn = ShaderStorageBuffer::new(
            bytemuck::cast_slice(&initial_nrm),
            RenderAssetUsages::RENDER_WORLD,
        );
        rn.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        ClothSimConfig {
            solve_substeps: SUBSTEPS,
            solve_inner_iterations: INNER_ITERS,
            collision_every_n_substeps: 1,
            coll_grid_origin,
            coll_grid_inv_cell,
            coll_grid_dims,
            coll_num_cells,
            coll_radix_digits,
            render_positions: buf_assets.add(rp),
            render_normals: buf_assets.add(rn),
            num_particles: self.num_particles,
            num_tris: (self.indices.len() / 3) as u32,
            num_distance_constraints: self.num_distance_constraints,
            constraint_batch_offsets: self.constraint_batch_offsets.clone(),
            constraint_batch_count: self.constraint_batch_count,
            constraint_i: self.constraint_i.clone(),
            constraint_j: self.constraint_j.clone(),
            constraint_rest_len: self.constraint_rest_len.clone(),
            constraint_compliance: self.constraint_compliance.clone(),
            tri_indices: self.indices.clone(),
            inv_mass: self.inv_mass.clone(),
            rest_pos,
            initial_pos,
        }
    }
}
