//! GPU cloth XPBD simulation (compute) + render graph dispatch.
//!
//! Distance constraints follow **XPBD** (Macklin et al., Eq. 17–18): α̃ = α/Δt², Δλ = (−C − α̃λ)/(∑ w + α̃),
//! Δx = M⁻¹∇Cᵀ Δλ. Jacobi clears **λ before each inner iteration** (parallel solves typically omit λ warm‑start
//! across Jacobi sweeps — carrying λ between sweeps often oscillates dense cloth). Edge pass (`jacobi_edges`)
//! plus gather (`jacobi_gather`). Compliance α comes from [`crate::mesh_prep`]
//! (`DEFAULT_STRETCH_COMPLIANCE` / `DEFAULT_BEND_COMPLIANCE`).
//!
//! **Stability:** Jacobi needs enough [`SUBSTEPS`] + [`INNER_ITERS`]; stretch α = 0 with few iterations often looks like edge-length “explosion”.
//! Tune [`ClothSimUniforms::jacobi_omega`] down if the sheet oscillates; raise [`JACOBI_CORRECTION_CAP`] only if edges are long (cap truncates corrections).
//!
//! **Note:** For the full write-up of what fixed dense-cloth blow-ups and the “ball” render artifact, see **`docs/CLOTH_SIM_STABILITY.md`** in this repo.

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
use std::sync::atomic::{AtomicU64, Ordering};

use crate::mesh_prep::ClothMeshData;

pub const CLOTH_SHADER: &str = "shaders/cloth_sim.wgsl";
/// XPBD (Müller et al.): use enough substeps so constraint corrections stay well-behaved vs. `dt`.
/// More substeps shrink substep `dt` → XPBD `α̃ = α/dt²` stays moderate and implicit integration is stabler.
pub const SUBSTEPS: u32 = 36;
/// Jacobi converges slower than Gauss–Seidel; extra iterations cut residual edge-length drift on free hems (ω stays low).
pub const INNER_ITERS: u32 = 22;
pub const DT: f32 = 1.0 / 60.0;
pub const THICKNESS: f32 = 0.04;
/// Scales overlap correction in `collide_pairs` / `collide_apply`; `0` disables self-collision.
pub const DEFAULT_COLL_SCALE: f32 = 0.38;
/// Linear air-drag coefficient \(k\) [1/s]: each substep applies `v *= exp(-k · Δt_sub)` in `post_velocity`
/// (matches `dv/dt ≈ -k v` for light viscous / laminar-like damping). **`0`** disables explicit drag.
///
/// The previous fixed factor-per-substep (~0.987) implied ~38% velocity loss **per frame**, far heavier than
/// typical cloth–air damping and made free motion sag slowly after releases.
pub const DEFAULT_LINEAR_AIR_DRAG_PER_SEC: f32 = 1.25;
/// Max length of one Jacobi position delta per inner iteration. Too small vs edge length → constraints never close → drift/explosion.
pub const JACOBI_CORRECTION_CAP: f32 = 0.28;
/// Hard speed clamp after gravity in `predict` (m/s).
pub const PREDICT_MAX_SPEED: f32 = 12.0;
/// Integer scale for GPU `atomicAdd` in `collide_pairs` (`cloth_sim.wgsl`); CPU sums `f32` directly.
pub const COLLISION_PAIR_FIXSCALE: i32 = 10_000;
/// Clamp on accumulated self-collision displacement per particle per substep (`collide_apply`).
pub const COLLISION_APPLY_CLAMP: f32 = 0.35;

/// Debug / tooling: pause GPU sim or advance one frame at a time (see example keyboard handler).
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
            sim_paused: true,
            step_serial: 0,
        }
    }
}

/// CPU-side config extracted to the render world.
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct ClothSimConfig {
    pub num_particles: u32,
    pub num_tris: u32,
    pub num_distance_constraints: u32,
    pub neighbor_packed: Vec<Vec4>,
    pub neighbor_offsets: Vec<u32>,
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
}

impl Default for ClothSimUniforms {
    fn default() -> Self {
        let sdt = DT / SUBSTEPS as f32;
        Self {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            num_particles: 0,
            num_tris: 0,
            // Parallel Jacobi + per-iteration λ clear needs modest ω; ~0.75+ often blows up dense cloth in frame 1.
            jacobi_omega: 0.38,
            inner_iterations: INNER_ITERS,
            thickness: THICKNESS,
            coll_scale: DEFAULT_COLL_SCALE,
            gravity: Vec4::new(0.0, -9.81, 0.0, 0.0),
            grab_target: Vec4::ZERO,
            grab_idx: -1,
            grab_active: 0,
            grab_stiffness: 0.45,
            floor_y: -0.7,
            linear_drag_per_sec: DEFAULT_LINEAR_AIR_DRAG_PER_SEC,
        }
    }
}

/// WGSL `SimParams` in `cloth_sim.wgsl` (`var<uniform>`). Written with [`bytemuck`] so GPU layout
/// cannot drift from `ClothSimUniforms` (encase) — kept in sync via unit test against `ShaderType::min_size`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClothSimParamsGpu {
    pub dt: f32,
    pub inv_dt: f32,
    pub num_particles: u32,
    pub num_tris: u32,
    pub jacobi_omega: f32,
    pub inner_iterations: u32,
    pub thickness: f32,
    pub coll_scale: f32,
    pub gravity: [f32; 4],
    pub grab_target: [f32; 4],
    pub grab_idx: i32,
    pub grab_active: u32,
    pub grab_stiffness: f32,
    pub floor_y: f32,
    pub linear_drag_per_sec: f32,
    /// WGSL uniform struct trailing alignment — zero-filled (`Pod`).
    pub _explicit_uniform_tail_padding: [u8; 12],
}

impl From<&ClothSimUniforms> for ClothSimParamsGpu {
    fn from(u: &ClothSimUniforms) -> Self {
        Self {
            dt: u.dt,
            inv_dt: u.inv_dt,
            num_particles: u.num_particles,
            num_tris: u.num_tris,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: u.inner_iterations,
            thickness: u.thickness,
            coll_scale: u.coll_scale,
            gravity: u.gravity.to_array(),
            grab_target: u.grab_target.to_array(),
            grab_idx: u.grab_idx,
            grab_active: u.grab_active,
            grab_stiffness: u.grab_stiffness,
            floor_y: u.floor_y,
            linear_drag_per_sec: u.linear_drag_per_sec,
            _explicit_uniform_tail_padding: [0u8; 12],
        }
    }
}

#[derive(Resource)]
pub struct ClothSimBuffers {
    /// `SimParams` (`var<uniform>`) — bytes from [`ClothSimParamsGpu`], not encase (layout-critical).
    pub params_uniform: Buffer,
    pub sim_pos: Buffer,
    pub jac_a: Buffer,
    pub jac_b: Buffer,
    pub prev: Buffer,
    pub vel: Buffer,
    pub rest: Buffer,
    pub inv_mass: Buffer,
    pub neigh_off: Buffer,
    pub neigh_pack: Buffer,
    pub constraint_i: Buffer,
    pub constraint_j: Buffer,
    pub constraint_rest: Buffer,
    pub constraint_comp: Buffer,
    pub constraint_lambda: Buffer,
    pub constraint_delta_lambda: Buffer,
    pub tri: Buffer,
    pub atomic_coll: Buffer,
    pub atomic_norm: Buffer,
}

#[derive(Resource)]
struct ClothPipeline {
    layout: BindGroupLayoutDescriptor,
    predict: CachedComputePipelineId,
    copy_sim_to_jac: CachedComputePipelineId,
    copy_jac_to_sim: CachedComputePipelineId,
    clear_constraint_lambda: CachedComputePipelineId,
    jacobi_edges: CachedComputePipelineId,
    jacobi_gather: CachedComputePipelineId,
    post_velocity: CachedComputePipelineId,
    clear_atomics: CachedComputePipelineId,
    collide_pairs: CachedComputePipelineId,
    collide_apply: CachedComputePipelineId,
    clear_norm_atomics: CachedComputePipelineId,
    accumulate_normals: CachedComputePipelineId,
    finalize_normals: CachedComputePipelineId,
}

#[derive(Resource)]
struct ClothBindGroups {
    /// predict, collide, post, normals: jac_in=A, jac_out=B (arbitrary; predict ignores jac_in)
    base: BindGroup,
    /// jacobi: read B, write A
    jac_b_to_a: BindGroup,
    /// jacobi: read A, write B
    jac_a_to_b: BindGroup,
    /// copy_jac_to_sim: read B -> sim
    copy_from_b: BindGroup,
}

#[derive(Resource, Default, Clone, Copy)]
enum ClothLoadState {
    #[default]
    Loading,
    Ready,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ClothSimLabel;

pub struct ClothComputePlugin;

impl Plugin for ClothComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClothSimControl>();
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
    let initial_params = ClothSimParamsGpu::from(uniforms.as_ref());
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
    let jac_a = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_jac_a"),
        size: vec4_sz(n),
        usage,
        mapped_at_creation: false,
    });
    let jac_b = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_jac_b"),
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
    let neigh_off = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_neigh_off"),
        size: u32_sz(n + 1),
        usage,
        mapped_at_creation: false,
    });
    let neigh_pack = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_neigh_pack"),
        size: vec4_sz(config.neighbor_packed.len()),
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

    render_queue.write_buffer(
        &neigh_pack,
        0,
        bytemuck::cast_slice::<Vec4, u8>(&config.neighbor_packed),
    );
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
    render_queue.write_buffer(
        &neigh_off,
        0,
        bytemuck::cast_slice::<u32, u8>(&config.neighbor_offsets),
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
    render_queue.write_buffer(&jac_a, 0, ip);
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
                storage_buffer_read_only_sized(false, None),
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
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
            ),
        ),
    );

    let shader = asset_server.load(CLOTH_SHADER);

    macro_rules! cp {
        ($name:expr) => {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                layout: vec![layout.clone()],
                shader: shader.clone(),
                entry_point: Some(Cow::from($name)),
                ..default()
            })
        };
    }

    commands.insert_resource(ClothPipeline {
        layout: layout.clone(),
        predict: cp!("predict"),
        copy_sim_to_jac: cp!("copy_sim_to_jac"),
        copy_jac_to_sim: cp!("copy_jac_to_sim"),
        clear_constraint_lambda: cp!("clear_constraint_lambda"),
        jacobi_edges: cp!("jacobi_edges"),
        jacobi_gather: cp!("jacobi_gather"),
        post_velocity: cp!("post_velocity"),
        clear_atomics: cp!("clear_atomics"),
        collide_pairs: cp!("collide_pairs"),
        collide_apply: cp!("collide_apply"),
        clear_norm_atomics: cp!("clear_norm_atomics"),
        accumulate_normals: cp!("accumulate_normals"),
        finalize_normals: cp!("finalize_normals"),
    });

    commands.insert_resource(ClothSimBuffers {
        params_uniform,
        sim_pos,
        jac_a,
        jac_b,
        prev,
        vel,
        rest,
        inv_mass: inv_mass_buf,
        neigh_off,
        neigh_pack,
        constraint_i: constraint_i_buf,
        constraint_j: constraint_j_buf,
        constraint_rest: constraint_rest_buf,
        constraint_comp: constraint_comp_buf,
        constraint_lambda: constraint_lambda_buf,
        constraint_delta_lambda: constraint_delta_lambda_buf,
        tri,
        atomic_coll,
        atomic_norm,
    });
}

fn make_bind_group(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    buffers: &ClothSimBuffers,
    gpu_rp: &GpuShaderStorageBuffer,
    gpu_rn: &GpuShaderStorageBuffer,
    jac_in: &Buffer,
    jac_out: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            buffers.params_uniform.as_entire_buffer_binding(),
            buffers.sim_pos.as_entire_binding(),
            jac_in.as_entire_buffer_binding(),
            jac_out.as_entire_buffer_binding(),
            buffers.prev.as_entire_binding(),
            buffers.vel.as_entire_binding(),
            buffers.rest.as_entire_binding(),
            buffers.inv_mass.as_entire_binding(),
            buffers.neigh_off.as_entire_binding(),
            buffers.neigh_pack.as_entire_binding(),
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

    let gpu_params = ClothSimParamsGpu::from(uniforms.as_ref());
    render_queue.write_buffer(
        &buffers.params_uniform,
        0,
        bytemuck::bytes_of(&gpu_params),
    );

    let base = make_bind_group(
        &render_device,
        &pipeline_cache,
        &pipeline.layout,
        &buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_a,
        &buffers.jac_b,
    );
    let jac_b_to_a = make_bind_group(
        &render_device,
        &pipeline_cache,
        &pipeline.layout,
        &buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_b,
        &buffers.jac_a,
    );
    let jac_a_to_b = make_bind_group(
        &render_device,
        &pipeline_cache,
        &pipeline.layout,
        &buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_a,
        &buffers.jac_b,
    );
    let copy_from_b = make_bind_group(
        &render_device,
        &pipeline_cache,
        &pipeline.layout,
        &buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_b,
        &buffers.jac_a,
    );

    commands.insert_resource(ClothBindGroups {
        base,
        jac_b_to_a,
        jac_a_to_b,
        copy_from_b,
    });
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
        pipeline.predict,
        pipeline.jacobi_edges,
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

        let n = config.num_particles;
        let wg64 = (n + 63) / 64;
        let wg256 = (n * 3 + 255) / 256;
        let wg_tris = (config.num_tris + 63) / 64;
        let pairs = n * (n - 1) / 2;
        let wg_pairs = (pairs + 255) / 256;
        let n_constraints = config.num_distance_constraints;
        let wg_constraints = (n_constraints + 63) / 64;

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        for _ in 0..SUBSTEPS {
            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(pipeline_cache.get_compute_pipeline(pipeline.predict).unwrap());
            pass.dispatch_workgroups(wg64, 1, 1);

            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.copy_sim_to_jac)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);

            if n_constraints > 0 {
                for k in 0..INNER_ITERS {
                    pass.set_bind_group(0, &bg.base, &[]);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.clear_constraint_lambda)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg_constraints, 1, 1);

                    let b = if k % 2 == 0 {
                        &bg.jac_b_to_a
                    } else {
                        &bg.jac_a_to_b
                    };
                    pass.set_bind_group(0, b, &[]);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.jacobi_edges)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg_constraints, 1, 1);

                    pass.set_bind_group(0, b, &[]);
                    pass.set_pipeline(
                        pipeline_cache
                            .get_compute_pipeline(pipeline.jacobi_gather)
                            .unwrap(),
                    );
                    pass.dispatch_workgroups(wg64, 1, 1);
                }
            }

            pass.set_bind_group(0, &bg.copy_from_b, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.copy_jac_to_sim)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);

            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.clear_atomics)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg256, 1, 1);

            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.collide_pairs)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg_pairs, 1, 1);

            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.collide_apply)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);

            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.post_velocity)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);
        }

        pass.set_bind_group(0, &bg.base, &[]);
        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(pipeline.clear_norm_atomics)
                .unwrap(),
        );
        pass.dispatch_workgroups(wg256, 1, 1);

        pass.set_bind_group(0, &bg.base, &[]);
        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(pipeline.accumulate_normals)
                .unwrap(),
        );
        pass.dispatch_workgroups(wg_tris, 1, 1);

        pass.set_bind_group(0, &bg.base, &[]);
        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(pipeline.finalize_normals)
                .unwrap(),
        );
        pass.dispatch_workgroups(wg64, 1, 1);

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
        assert_eq!(
            ClothSimUniforms::min_size().get(),
            std::mem::size_of::<ClothSimParamsGpu>() as u64,
            "hand-written GPU struct size must match encase/uniform layout"
        );
    }
}

impl ClothMeshData {
    pub fn to_sim_config(
        &self,
        buf_assets: &mut Assets<ShaderStorageBuffer>,
    ) -> ClothSimConfig {
        let n = self.num_particles as usize;
        let mut packed = Vec::new();
        for pi in 0..n {
            let s = self.neighbor_offsets[pi] as usize;
            let e = self.neighbor_offsets[pi + 1] as usize;
            for k in s..e {
                packed.push(Vec4::new(
                    self.neighbor_other[k] as f32,
                    self.neighbor_rest_len[k],
                    self.neighbor_compliance[k],
                    self.neighbor_constraint_id[k] as f32,
                ));
            }
        }

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
            render_positions: buf_assets.add(rp),
            render_normals: buf_assets.add(rn),
            num_particles: self.num_particles,
            num_tris: (self.indices.len() / 3) as u32,
            num_distance_constraints: self.num_distance_constraints,
            neighbor_packed: packed,
            neighbor_offsets: self.neighbor_offsets.clone(),
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
