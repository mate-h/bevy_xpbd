//! GPU Jacobi distance solver path (default; enable `solver-gauss-seidel` for colored GS).

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderDevice, RenderQueue},
        storage::GpuShaderBuffer,
    },
    shader::ShaderCacheError,
};
use std::borrow::Cow;
use std::num::NonZero;

use crate::cloth_compute::{
    ClothCollGridGpu, ClothCollRadixPassGpu, ClothLoadState, ClothSimConfig, ClothSimControl,
    ClothSimParamsGpu, ClothSimUniforms, CLOTH_SHADER_JACOBI,
};
use crate::mesh_prep::ClothMeshData;

#[derive(Resource)]
pub struct ClothSimBuffersJacobi {
    pub params_uniform: Buffer,
    pub sim_pos: Buffer,
    pub jac_a: Buffer,
    pub jac_b: Buffer,
    pub prev: Buffer,
    pub vel: Buffer,
    pub rest: Buffer,
    pub inv_mass: Buffer,
    pub neighbor_offsets: Buffer,
    pub neighbor_packed: Buffer,
    pub constraint_i: Buffer,
    pub constraint_j: Buffer,
    pub constraint_rest: Buffer,
    pub constraint_comp: Buffer,
    pub constraint_lambda: Buffer,
    pub constraint_delta_lambda: Buffer,
    pub tri: Buffer,
    pub atomic_coll: Buffer,
    pub atomic_norm: Buffer,
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
pub struct ClothPipelineJacobi {
    pub layout: BindGroupLayoutDescriptor,
    pub predict_copy_sim_to_jac: CachedComputePipelineId,
    pub copy_jac_to_sim: CachedComputePipelineId,
    pub clear_constraint_lambda: CachedComputePipelineId,
    pub jacobi_edges: CachedComputePipelineId,
    pub jacobi_gather: CachedComputePipelineId,
    pub post_velocity: CachedComputePipelineId,
    pub clear_atomics: CachedComputePipelineId,
    pub coll_cell_bounds_clear: CachedComputePipelineId,
    pub coll_perm_identity_ping: CachedComputePipelineId,
    pub coll_histogram_clear: CachedComputePipelineId,
    pub coll_radix_digit_count: CachedComputePipelineId,
    pub coll_radix_exclusive_bases_heads: CachedComputePipelineId,
    pub coll_radix_digit_scatter: CachedComputePipelineId,
    pub coll_sorted_build_cell_ranges: CachedComputePipelineId,
    pub collide_grid_cells: CachedComputePipelineId,
    pub collide_apply: CachedComputePipelineId,
    pub clear_norm_atomics: CachedComputePipelineId,
    pub accumulate_normals: CachedComputePipelineId,
    pub finalize_normals: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct ClothBindGroupsJacobi {
    /// predict, collision tail, post_velocity, normals: jac_in=A, jac_out=B
    pub base: BindGroup,
    /// jacobi inner: read B, write A
    pub jac_b_to_a: BindGroup,
    /// jacobi inner: read A, write B
    pub jac_a_to_b: BindGroup,
    /// copy_jac_to_sim when final jac buffer is A
    pub copy_from_a: BindGroup,
    /// copy_jac_to_sim when final jac buffer is B
    pub copy_from_b: BindGroup,
}

pub fn pack_neighbor_gpu(mesh: &ClothMeshData) -> Vec<Vec4> {
    let n = mesh.num_particles as usize;
    let mut packed = Vec::new();
    for pi in 0..n {
        let s = mesh.neighbor_offsets[pi] as usize;
        let e = mesh.neighbor_offsets[pi + 1] as usize;
        for k in s..e {
            packed.push(Vec4::new(
                mesh.neighbor_other[k] as f32,
                mesh.neighbor_rest_len[k],
                mesh.neighbor_compliance[k],
                mesh.neighbor_constraint_id[k] as f32,
            ));
        }
    }
    packed
}

fn jacobi_bind_layout(
    radix_nz: NonZero<u64>,
    perm_nz: NonZero<u64>,
    cell_nz: NonZero<u64>,
) -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "cloth_sim_jacobi",
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
    )
}

fn make_jacobi_bind_group(
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    buffers: &ClothSimBuffersJacobi,
    gpu_rp: &GpuShaderBuffer,
    gpu_rn: &GpuShaderBuffer,
    jac_in: &Buffer,
    jac_out: &Buffer,
) -> BindGroup {
    render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(layout),
        &BindGroupEntries::sequential((
            buffers.params_uniform.as_entire_buffer_binding(),
            buffers.sim_pos.as_entire_binding(),
            jac_in.as_entire_binding(),
            jac_out.as_entire_binding(),
            buffers.prev.as_entire_binding(),
            buffers.vel.as_entire_binding(),
            buffers.rest.as_entire_binding(),
            buffers.inv_mass.as_entire_binding(),
            buffers.neighbor_offsets.as_entire_binding(),
            buffers.neighbor_packed.as_entire_binding(),
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

pub fn init_cloth_sim_jacobi(
    commands: &mut Commands,
    config: &ClothSimConfig,
    uniforms: &ClothSimUniforms,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
) {
    let n = config.num_particles as usize;
    let n3 = n * 3;
    let vec4_sz = |count: usize| (count * 16) as u64;
    let f32_sz = |count: usize| (count * 4) as u64;
    let u32_sz = |count: usize| (count * 4) as u64;
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    let params_uniform = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_sim_params_uniform"),
        size: std::mem::size_of::<ClothSimParamsGpu>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let initial_params = ClothSimParamsGpu::pack(uniforms, 0);
    render_queue.write_buffer(&params_uniform, 0, bytemuck::bytes_of(&initial_params));

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
    let neigh_off_len = config.neighbor_offsets.len().max(2);
    let neighbor_offsets = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_neighbor_offsets"),
        size: u32_sz(neigh_off_len),
        usage,
        mapped_at_creation: false,
    });
    let np_len = config.neighbor_packed.len().max(1);
    let neighbor_packed = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_neighbor_packed"),
        size: vec4_sz(np_len),
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

    let coll_grid_u_sz = std::mem::size_of::<ClothCollGridGpu>() as u64;
    let coll_radix_pass_sz = std::mem::size_of::<ClothCollRadixPassGpu>() as u64;
    let radix_arr = 256u64 * 4;
    let radix_nz = NonZero::new(radix_arr).expect("radix");
    let nc = config.coll_num_cells.max(1) as usize;
    let perm_nz = NonZero::new(u32_sz(n)).expect("perm");
    let cell_nz = NonZero::new(u32_sz(nc)).expect("cell");

    let coll_grid_uniform = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloth_coll_grid_uniform"),
        size: coll_grid_u_sz,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(
        &coll_grid_uniform,
        0,
        bytemuck::bytes_of(&ClothCollGridGpu::pack_from_config(config)),
    );

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
    let mut neigh_off_upload = config.neighbor_offsets.clone();
    neigh_off_upload.resize(neigh_off_len, 0);
    render_queue.write_buffer(
        &neighbor_offsets,
        0,
        bytemuck::cast_slice::<u32, u8>(&neigh_off_upload),
    );
    if !config.neighbor_packed.is_empty() {
        render_queue.write_buffer(
            &neighbor_packed,
            0,
            bytemuck::cast_slice::<Vec4, u8>(&config.neighbor_packed),
        );
    }
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
    render_queue.write_buffer(&rest, 0, bytemuck::cast_slice::<Vec4, u8>(&config.rest_pos));
    let ip = bytemuck::cast_slice::<Vec4, u8>(&config.initial_pos);
    render_queue.write_buffer(&sim_pos, 0, ip);
    render_queue.write_buffer(&jac_a, 0, ip);
    render_queue.write_buffer(&jac_b, 0, ip);
    render_queue.write_buffer(&prev, 0, ip);
    render_queue.write_buffer(&vel, 0, &vec![0u8; vec4_sz(n) as usize]);

    let layout = jacobi_bind_layout(radix_nz, perm_nz, cell_nz);
    let shader = asset_server.load(CLOTH_SHADER_JACOBI);

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

    commands.insert_resource(ClothPipelineJacobi {
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
        jacobi_edges: cp!("cloth_cs_jacobi_edges", "jacobi_edges"),
        jacobi_gather: cp!("cloth_cs_jacobi_gather", "jacobi_gather"),
        post_velocity: cp!("cloth_cs_post_velocity", "post_velocity"),
        clear_atomics: cp!("cloth_cs_clear_atomics", "clear_atomics"),
        coll_cell_bounds_clear: cp!("cloth_cs_coll_cell_bounds_clear", "coll_cell_bounds_clear"),
        coll_perm_identity_ping: cp!(
            "cloth_cs_coll_perm_identity_ping",
            "coll_perm_identity_ping"
        ),
        coll_histogram_clear: cp!("cloth_cs_coll_histogram_clear", "coll_histogram_clear"),
        coll_radix_digit_count: cp!("cloth_cs_coll_radix_digit_count", "coll_radix_digit_count"),
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

    commands.insert_resource(ClothSimBuffersJacobi {
        params_uniform,
        sim_pos,
        jac_a,
        jac_b,
        prev,
        vel,
        rest,
        inv_mass: inv_mass_buf,
        neighbor_offsets,
        neighbor_packed,
        constraint_i: constraint_i_buf,
        constraint_j: constraint_j_buf,
        constraint_rest: constraint_rest_buf,
        constraint_comp: constraint_comp_buf,
        constraint_lambda: constraint_lambda_buf,
        constraint_delta_lambda: constraint_delta_lambda_buf,
        tri,
        atomic_coll,
        atomic_norm,
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

pub fn prepare_cloth_bind_groups_jacobi(
    commands: &mut Commands,
    pipeline: &ClothPipelineJacobi,
    buffers: &ClothSimBuffersJacobi,
    uniforms: &ClothSimUniforms,
    config: &ClothSimConfig,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    pipeline_cache: &PipelineCache,
    gpu_rp: &GpuShaderBuffer,
    gpu_rn: &GpuShaderBuffer,
) {
    let gpu_params = ClothSimParamsGpu::pack(uniforms, 0);
    render_queue.write_buffer(&buffers.params_uniform, 0, bytemuck::bytes_of(&gpu_params));
    render_queue.write_buffer(
        &buffers.coll_grid_uniform,
        0,
        bytemuck::bytes_of(&ClothCollGridGpu::pack_from_config(config)),
    );

    let base = make_jacobi_bind_group(
        render_device,
        pipeline_cache,
        &pipeline.layout,
        buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_a,
        &buffers.jac_b,
    );
    let jac_b_to_a = make_jacobi_bind_group(
        render_device,
        pipeline_cache,
        &pipeline.layout,
        buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_b,
        &buffers.jac_a,
    );
    let jac_a_to_b = make_jacobi_bind_group(
        render_device,
        pipeline_cache,
        &pipeline.layout,
        buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_a,
        &buffers.jac_b,
    );
    let copy_from_a = make_jacobi_bind_group(
        render_device,
        pipeline_cache,
        &pipeline.layout,
        buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_a,
        &buffers.jac_b,
    );
    let copy_from_b = make_jacobi_bind_group(
        render_device,
        pipeline_cache,
        &pipeline.layout,
        buffers,
        gpu_rp,
        gpu_rn,
        &buffers.jac_b,
        &buffers.jac_a,
    );

    commands.insert_resource(ClothBindGroupsJacobi {
        base,
        jac_b_to_a,
        jac_a_to_b,
        copy_from_a,
        copy_from_b,
    });
}

pub fn check_cloth_pipeline_jacobi(
    pipeline_cache: &PipelineCache,
    pipeline: &ClothPipelineJacobi,
    state: &mut ClothLoadState,
) {
    if matches!(*state, ClothLoadState::Ready) {
        return;
    }
    for id in [
        pipeline.predict_copy_sim_to_jac,
        pipeline.jacobi_edges,
        pipeline.collide_grid_cells,
        pipeline.finalize_normals,
    ] {
        match pipeline_cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Ok(_) => {}
            CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => return,
            CachedPipelineState::Err(e) => panic!("cloth jacobi shader: {e}"),
            _ => return,
        }
    }
    *state = ClothLoadState::Ready;
}

pub fn run_cloth_sim_jacobi(
    encoder: &mut CommandEncoder,
    bg: &ClothBindGroupsJacobi,
    pipeline: &ClothPipelineJacobi,
    pipeline_cache: &PipelineCache,
    buffers: &ClothSimBuffersJacobi,
    config: &ClothSimConfig,
    uniforms: &ClothSimUniforms,
    render_queue: &RenderQueue,
    last_ack_step_serial: &mut u64,
    ctrl: &ClothSimControl,
) {
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
    let substeps = config.solve_substeps.max(1) as usize;
    let inner_iters = config.solve_inner_iterations.max(1) as usize;
    let coll_tail_enabled = uniforms.coll_scale > 1e-12;
    let col_stride = config.collision_every_n_substeps.max(1) as usize;

    for si in 0..substeps {
        let run_collision_trio =
            coll_tail_enabled && (si % col_stride == col_stride.saturating_sub(1));

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth_pass_integrate_jac_seed"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg.base, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.predict_copy_sim_to_jac)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);
        }

        if n_constraints > 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth_pass_distance_jacobi"),
                timestamp_writes: None,
            });
            for k in 0..inner_iters {
                let ping = if k % 2 == 0 {
                    &bg.jac_b_to_a
                } else {
                    &bg.jac_a_to_b
                };
                pass.set_bind_group(0, ping, &[]);
                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.clear_constraint_lambda)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg_constraints, 1, 1);

                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.jacobi_edges)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg_constraints, 1, 1);

                pass.set_pipeline(
                    pipeline_cache
                        .get_compute_pipeline(pipeline.jacobi_gather)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg64, 1, 1);
            }
        }

        let copy_bg = if inner_iters % 2 == 0 {
            &bg.copy_from_b
        } else {
            &bg.copy_from_a
        };

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloth_pass_collision_velocity"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, copy_bg, &[]);
            pass.set_pipeline(
                pipeline_cache
                    .get_compute_pipeline(pipeline.copy_jac_to_sim)
                    .unwrap(),
            );
            pass.dispatch_workgroups(wg64, 1, 1);

            if run_collision_trio {
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
                        .get_compute_pipeline(pipeline.coll_perm_identity_ping)
                        .unwrap(),
                );
                pass.dispatch_workgroups(wg_n256.max(1), 1, 1);
            }
        }

        if run_collision_trio {
            for d in 0..radix_digits {
                let radix_u = ClothCollRadixPassGpu { data: [d, 0, 0, 0] };
                render_queue.write_buffer(
                    &buffers.coll_radix_pass_uniform,
                    0,
                    bytemuck::bytes_of(&radix_u),
                );

                let mut rpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloth_pass_collision_radix"),
                    timestamp_writes: None,
                });
                rpass.set_bind_group(0, &bg.base, &[]);
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
            cpass.set_bind_group(0, &bg.base, &[]);
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
            pass.set_bind_group(0, &bg.base, &[]);
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
    }

    *last_ack_step_serial = ctrl.step_serial;
}

pub fn jacobi_default_omega() -> f32 {
    0.32
}

/// Jacobi converges slower per iteration than colored GS — default mesh config uses this when `solver-jacobi` is enabled.
pub fn jacobi_default_inner_iters() -> u32 {
    22
}
