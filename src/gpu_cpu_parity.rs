//! Compare one GPU substep (raw `wgpu` dispatch of `cloth_sim.wgsl`) to [`crate::xpbd_cpu`].

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::path::Path;

use bevy::math::{Vec3, Vec4};

use crate::cloth_compute::{
    ClothCollGridGpu, ClothCollRadixPassGpu, ClothSimParamsGpu, ClothSimUniforms,
    GS_BATCH_DYNAMIC_STRIDE, GS_EDGE_THREADS, INNER_ITERS, REFERENCE_FRAME_DELTA_SECS, SUBSTEPS,
    THICKNESS,
};
use crate::mesh_prep::ClothMeshData;
use crate::xpbd_cpu::{xpbd_substep_with_self_collision, XpbdCpuTimeStepParams};

fn vec4_buf(n: usize) -> u64 {
    (n * 16) as u64
}

struct WgpuClothContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    predict_copy_sim_to_jac: wgpu::ComputePipeline,
    copy_jac_to_sim: wgpu::ComputePipeline,
    clear_constraint_lambda: wgpu::ComputePipeline,
    gs_edges: wgpu::ComputePipeline,
    post_velocity: wgpu::ComputePipeline,
    clear_atomics: wgpu::ComputePipeline,
    coll_cell_bounds_clear: wgpu::ComputePipeline,
    coll_perm_identity_ping: wgpu::ComputePipeline,
    coll_histogram_clear: wgpu::ComputePipeline,
    coll_radix_digit_count: wgpu::ComputePipeline,
    coll_radix_exclusive_bases_heads: wgpu::ComputePipeline,
    coll_radix_digit_scatter: wgpu::ComputePipeline,
    coll_sorted_build_cell_ranges: wgpu::ComputePipeline,
    collide_grid_cells: wgpu::ComputePipeline,
    collide_apply: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
}

impl WgpuClothContext {
    fn new(instance: &wgpu::Instance) -> Option<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cloth_parity"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: Default::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/cloth_sim.wgsl");
        let source = std::fs::read_to_string(&shader_path).ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloth_sim"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });

        let u_size = NonZeroU64::new(std::mem::size_of::<ClothSimParamsGpu>() as u64).unwrap();
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloth_sim"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(u_size),
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
                storage_entry(9, true),
                storage_entry(10, true),
                storage_entry(11, true),
                storage_entry(12, false),
                storage_entry(13, false),
                storage_entry(14, true),
                storage_entry(15, false),
                storage_entry(16, false),
                storage_entry(17, false),
                storage_entry(18, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(GS_BATCH_DYNAMIC_STRIDE as u64),
                    },
                    count: None,
                },
                uniform_fixed_entry(20, std::mem::size_of::<ClothCollGridGpu>() as u64),
                uniform_fixed_entry(21, std::mem::size_of::<ClothCollRadixPassGpu>() as u64),
                storage_sized_entry(22, false, 256 * 4),
                storage_sized_entry(23, false, 256 * 4),
                storage_entry(24, false),
                storage_entry(25, false),
                storage_entry(26, false),
                storage_entry(27, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloth_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        macro_rules! cp {
            ($entry:literal) => {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(concat!("cloth_cs_", $entry)),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some($entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
        }

        Some(Self {
            predict_copy_sim_to_jac: cp!("predict_copy_sim_to_jac"),
            copy_jac_to_sim: cp!("copy_jac_to_sim"),
            clear_constraint_lambda: cp!("clear_constraint_lambda"),
            gs_edges: cp!("gs_edges"),
            post_velocity: cp!("post_velocity"),
            clear_atomics: cp!("clear_atomics"),
            coll_cell_bounds_clear: cp!("coll_cell_bounds_clear"),
            coll_perm_identity_ping: cp!("coll_perm_identity_ping"),
            coll_histogram_clear: cp!("coll_histogram_clear"),
            coll_radix_digit_count: cp!("coll_radix_digit_count"),
            coll_radix_exclusive_bases_heads: cp!("coll_radix_exclusive_bases_heads"),
            coll_radix_digit_scatter: cp!("coll_radix_digit_scatter"),
            coll_sorted_build_cell_ranges: cp!("coll_sorted_build_cell_ranges"),
            collide_grid_cells: cp!("collide_grid_cells"),
            collide_apply: cp!("collide_apply"),
            device,
            queue,
            bind_layout,
        })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_fixed_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn storage_sized_entry(binding: u32, read_only: bool, bytes: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(bytes),
        },
        count: None,
    }
}

struct Buffers {
    params: wgpu::Buffer,
    sim_pos: wgpu::Buffer,
    jac_state: wgpu::Buffer,
    prev: wgpu::Buffer,
    vel: wgpu::Buffer,
    rest: wgpu::Buffer,
    inv_mass: wgpu::Buffer,
    constraint_batch_offsets: wgpu::Buffer,
    constraint_i: wgpu::Buffer,
    constraint_j: wgpu::Buffer,
    constraint_rest: wgpu::Buffer,
    constraint_comp: wgpu::Buffer,
    constraint_lambda: wgpu::Buffer,
    constraint_delta_lambda: wgpu::Buffer,
    tri: wgpu::Buffer,
    render_pos: wgpu::Buffer,
    render_nrm: wgpu::Buffer,
    atomic_coll: wgpu::Buffer,
    atomic_norm: wgpu::Buffer,
    gs_batch_dyn: wgpu::Buffer,
    coll_grid_uniform: wgpu::Buffer,
    coll_radix_pass_uniform: wgpu::Buffer,
    coll_radix_hist: wgpu::Buffer,
    coll_radix_head: wgpu::Buffer,
    coll_perm_ping: wgpu::Buffer,
    coll_perm_pong: wgpu::Buffer,
    coll_cell_start: wgpu::Buffer,
    coll_cell_end_exclusive: wgpu::Buffer,
    coll_num_cells: u32,
    coll_radix_digits: u32,
}

fn make_buffers(ctx: &WgpuClothContext, mesh: &ClothMeshData) -> Buffers {
    let dev = &ctx.device;
    let n = mesh.num_particles as usize;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let initial_pos: Vec<Vec4> = mesh
        .positions
        .iter()
        .map(|p| Vec4::new(p.x, p.y, p.z, 0.0))
        .collect();
    let rest_pos: Vec<Vec4> = mesh
        .rest_positions
        .iter()
        .map(|p| Vec4::new(p.x, p.y, p.z, 0.0))
        .collect();
    let initial_nrm: Vec<Vec4> = mesh
        .normals
        .iter()
        .map(|v| Vec4::new(v.x, v.y, v.z, 0.0))
        .collect();

    let vb = |label: &'static str, size: u64| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };

    let sim_pos = vb("cloth_sim_pos", vec4_buf(n));
    let jac_state = vb("cloth_jac_state", vec4_buf(n));
    let prev = vb("cloth_prev_pos", vec4_buf(n));
    let vel = vb("cloth_velocity", vec4_buf(n));
    let rest = vb("cloth_rest_pos", vec4_buf(n));
    let render_pos = vb("cloth_render_positions", vec4_buf(n));
    let render_nrm = vb("cloth_render_normals", vec4_buf(n));

    let ip = bytemuck::cast_slice::<Vec4, u8>(&initial_pos);
    ctx.queue.write_buffer(&sim_pos, 0, ip);
    ctx.queue.write_buffer(&jac_state, 0, ip);
    ctx.queue.write_buffer(&prev, 0, ip);
    ctx.queue
        .write_buffer(&rest, 0, bytemuck::cast_slice::<Vec4, u8>(&rest_pos));
    ctx.queue
        .write_buffer(&vel, 0, &vec![0u8; vec4_buf(n) as usize]);
    ctx.queue.write_buffer(
        &render_pos,
        0,
        bytemuck::cast_slice::<Vec4, u8>(&initial_pos),
    );
    ctx.queue.write_buffer(
        &render_nrm,
        0,
        bytemuck::cast_slice::<Vec4, u8>(&initial_nrm),
    );

    let inv_mass = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_inv_mass"),
        size: (n * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &inv_mass,
        0,
        bytemuck::cast_slice::<f32, u8>(&mesh.inv_mass),
    );

    let bo_len = mesh.constraint_batch_offsets.len().max(2);
    let mut batch_offs_upload = mesh.constraint_batch_offsets.clone();
    batch_offs_upload.resize(bo_len, 0);

    let constraint_batch_offsets = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_batch_offsets"),
        size: (bo_len * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &constraint_batch_offsets,
        0,
        bytemuck::cast_slice::<u32, u8>(&batch_offs_upload),
    );

    let ec = mesh.num_distance_constraints as usize;
    let ec_store = ec.max(1);
    let constraint_i = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_i"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_j = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_j"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_rest = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_rest"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_comp = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_comp"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_lambda = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_lambda"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_delta_lambda = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_constraint_delta_lambda"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    if ec > 0 {
        ctx.queue.write_buffer(
            &constraint_i,
            0,
            bytemuck::cast_slice::<u32, u8>(&mesh.constraint_i),
        );
        ctx.queue.write_buffer(
            &constraint_j,
            0,
            bytemuck::cast_slice::<u32, u8>(&mesh.constraint_j),
        );
        ctx.queue.write_buffer(
            &constraint_rest,
            0,
            bytemuck::cast_slice::<f32, u8>(&mesh.constraint_rest_len),
        );
        ctx.queue.write_buffer(
            &constraint_comp,
            0,
            bytemuck::cast_slice::<f32, u8>(&mesh.constraint_compliance),
        );
    }
    ctx.queue
        .write_buffer(&constraint_lambda, 0, &vec![0u8; ec_store * 4]);
    ctx.queue
        .write_buffer(&constraint_delta_lambda, 0, &vec![0u8; ec_store * 4]);

    let tri = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_tri_indices"),
        size: (mesh.indices.len() * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&tri, 0, bytemuck::cast_slice::<u32, u8>(&mesh.indices));

    let n3 = (n * 3 * 4) as u64;
    let atomic_coll = vb("cloth_atomic_coll", n3);
    let atomic_norm = vb("cloth_atomic_norm", n3);

    let nb_lut = mesh.constraint_batch_count.max(1) as usize;
    let gs_dyn_bytes = GS_BATCH_DYNAMIC_STRIDE as usize * nb_lut;
    let mut gs_dyn_lut = vec![0u8; gs_dyn_bytes];
    for bat in 0..(mesh.constraint_batch_count as usize) {
        let o = bat * GS_BATCH_DYNAMIC_STRIDE as usize;
        gs_dyn_lut[o..o + 4].copy_from_slice(&(bat as u32).to_le_bytes());
    }
    let gs_batch_dyn = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_gs_batch_dyn"),
        size: gs_dyn_bytes as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&gs_batch_dyn, 0, &gs_dyn_lut);

    let (grid_origin, coll_inv_cell, coll_dims, coll_nc, coll_rd) =
        crate::mesh_prep::derive_collision_grid(&rest_pos, THICKNESS);
    let coll_num_cells_meta = coll_nc.max(1);
    let coll_radix_digits_meta = coll_rd.max(1);
    let nc_usize = coll_num_cells_meta as usize;

    let coll_grid_uniform = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_coll_grid_uniform"),
        size: std::mem::size_of::<ClothCollGridGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let gpu_grid = ClothCollGridGpu {
        grid_origin_pad: [grid_origin.x, grid_origin.y, grid_origin.z, 0.0],
        inv_cell: coll_inv_cell,
        num_cells: coll_num_cells_meta,
        num_particles: mesh.num_particles,
        gx: coll_dims[0],
        gy: coll_dims[1],
        gz: coll_dims[2],
        radix_digits: coll_radix_digits_meta,
        _align_pad: [0u8; 4],
        _reserved: [0u32; 4],
    };
    ctx.queue
        .write_buffer(&coll_grid_uniform, 0, bytemuck::bytes_of(&gpu_grid));

    let coll_radix_pass_uniform = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_coll_radix_pass_uniform"),
        size: std::mem::size_of::<ClothCollRadixPassGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &coll_radix_pass_uniform,
        0,
        bytemuck::bytes_of(&ClothCollRadixPassGpu { data: [0u32; 4] }),
    );

    let radix_sz = (256 * 4) as u64;
    let coll_radix_hist = vb("cloth_coll_radix_hist", radix_sz);
    let coll_radix_head = vb("cloth_coll_radix_head", radix_sz);
    let coll_perm_ping = vb("cloth_coll_perm_ping", (n * 4) as u64);
    let coll_perm_pong = vb("cloth_coll_perm_pong", (n * 4) as u64);
    let coll_cell_start = vb(
        "cloth_coll_cell_start",
        (std::mem::size_of::<u32>() as u64).saturating_mul(nc_usize as u64),
    );
    let coll_cell_end_exclusive = vb(
        "cloth_coll_cell_end_exclusive",
        (std::mem::size_of::<u32>() as u64).saturating_mul(nc_usize as u64),
    );

    let params = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_sim_params_uniform"),
        size: std::mem::size_of::<ClothSimParamsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Buffers {
        params,
        sim_pos,
        jac_state,
        prev,
        vel,
        rest,
        inv_mass,
        constraint_batch_offsets,
        constraint_i,
        constraint_j,
        constraint_rest,
        constraint_comp,
        constraint_lambda,
        constraint_delta_lambda,
        tri,
        render_pos,
        render_nrm,
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
        coll_num_cells: coll_num_cells_meta,
        coll_radix_digits: coll_radix_digits_meta,
    }
}

fn make_bind_group(ctx: &WgpuClothContext, b: &Buffers) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &ctx.bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b.params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.sim_pos.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.jac_state.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b.prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b.vel.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b.rest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b.inv_mass.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: b.constraint_batch_offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: b.constraint_i.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: b.constraint_j.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: b.constraint_rest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: b.constraint_comp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: b.constraint_lambda.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: b.constraint_delta_lambda.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: b.tri.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: b.render_pos.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: b.render_nrm.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: b.atomic_coll.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 18,
                resource: b.atomic_norm.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 19,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &b.gs_batch_dyn,
                    offset: 0,
                    size: wgpu::BufferSize::new(GS_BATCH_DYNAMIC_STRIDE.into()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 20,
                resource: b.coll_grid_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 21,
                resource: b.coll_radix_pass_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 22,
                resource: b.coll_radix_hist.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 23,
                resource: b.coll_radix_head.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 24,
                resource: b.coll_perm_ping.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 25,
                resource: b.coll_perm_pong.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 26,
                resource: b.coll_cell_start.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 27,
                resource: b.coll_cell_end_exclusive.as_entire_binding(),
            },
        ],
    })
}

fn run_one_gpu_substep(
    ctx: &WgpuClothContext,
    b: &Buffers,
    bg: &wgpu::BindGroup,
    mesh: &ClothMeshData,
    n: u32,
    num_constraints: u32,
) {
    let wg64 = ((n as usize) + 63) / 64;
    let wg256 = (n as usize * 3 + 255) / 256;
    let wg_n256_parity = (((n as usize) + 255) / 256).max(1);
    let num_cells_b = b.coll_num_cells.max(1);
    let wg_cell_clear = ((num_cells_b as usize) + 255) / 256;
    let radix_digits = b.coll_radix_digits.max(1);
    let wg_constraints = (num_constraints as usize + 63) / 64;
    let num_batches = mesh.constraint_batch_count as usize;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloth_encoder_substep"),
        });
    const DYN_IDLE: &[u32] = &[0];

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_integrate_jac_seed"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.predict_copy_sim_to_jac);
        pass.set_bind_group(0, bg, DYN_IDLE);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);
    }

    if num_constraints > 0 && mesh.constraint_batch_count > 0 {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_distance_gauss_seidel"),
            timestamp_writes: None,
        });
        for _k in 0..INNER_ITERS {
            pass.set_pipeline(&ctx.clear_constraint_lambda);
            pass.set_bind_group(0, bg, DYN_IDLE);
            pass.dispatch_workgroups(wg_constraints as u32, 1, 1);

            pass.set_pipeline(&ctx.gs_edges);
            for bat in 0..num_batches {
                let start = mesh.constraint_batch_offsets[bat] as usize;
                let end = mesh.constraint_batch_offsets[bat + 1] as usize;
                let span = end.saturating_sub(start);
                if span == 0 {
                    continue;
                }
                let t = GS_EDGE_THREADS.max(1) as usize;
                let wg_batch = ((span + (t - 1)) / t) as u32;
                let dyn_off = (bat as u32).saturating_mul(GS_BATCH_DYNAMIC_STRIDE);
                pass.set_bind_group(0, bg, &[dyn_off]);
                pass.dispatch_workgroups(wg_batch, 1, 1);
            }
        }
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_collision_velocity_seed"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.copy_jac_to_sim);
        pass.set_bind_group(0, bg, DYN_IDLE);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);

        pass.set_pipeline(&ctx.clear_atomics);
        pass.set_bind_group(0, bg, DYN_IDLE);
        pass.dispatch_workgroups(wg256 as u32, 1, 1);

        pass.set_pipeline(&ctx.coll_perm_identity_ping);
        pass.set_bind_group(0, bg, DYN_IDLE);
        pass.dispatch_workgroups(wg_n256_parity as u32, 1, 1);
    }

    for d in 0..radix_digits {
        let radix_u = ClothCollRadixPassGpu { data: [d, 0, 0, 0] };
        ctx.queue
            .write_buffer(&b.coll_radix_pass_uniform, 0, bytemuck::bytes_of(&radix_u));

        let mut rpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_collision_radix"),
            timestamp_writes: None,
        });
        rpass.set_bind_group(0, bg, DYN_IDLE);
        rpass.set_pipeline(&ctx.coll_histogram_clear);
        rpass.dispatch_workgroups(1, 1, 1);

        rpass.set_pipeline(&ctx.coll_radix_digit_count);
        rpass.dispatch_workgroups(wg_n256_parity as u32, 1, 1);

        rpass.set_pipeline(&ctx.coll_radix_exclusive_bases_heads);
        rpass.dispatch_workgroups(1, 1, 1);

        rpass.set_pipeline(&ctx.coll_radix_digit_scatter);
        rpass.dispatch_workgroups(wg_n256_parity as u32, 1, 1);
    }

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_collision_grid_narrow"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, bg, DYN_IDLE);
        cpass.set_pipeline(&ctx.coll_cell_bounds_clear);
        cpass.dispatch_workgroups(wg_cell_clear.max(1) as u32, 1, 1);

        cpass.set_pipeline(&ctx.coll_sorted_build_cell_ranges);
        cpass.dispatch_workgroups(wg_n256_parity as u32, 1, 1);

        cpass.set_pipeline(&ctx.collide_grid_cells);
        cpass.dispatch_workgroups(wg_n256_parity as u32, 1, 1);

        cpass.set_pipeline(&ctx.collide_apply);
        cpass.dispatch_workgroups(wg64 as u32, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloth_pass_post_velocity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.post_velocity);
        pass.set_bind_group(0, bg, DYN_IDLE);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

fn read_vec4_positions(ctx: &WgpuClothContext, buf: &wgpu::Buffer, n: usize) -> Vec<Vec3> {
    let size = vec4_buf(n);
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cloth_buffer_readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloth_encoder_copy_positions"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice.get_mapped_range();
    let flat: Vec<Vec4> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    flat.into_iter().map(|v| Vec3::new(v.x, v.y, v.z)).collect()
}

fn triangle_cloth() -> ClothMeshData {
    let obj = r#"
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
"#;
    crate::mesh_prep::parse_welded_obj(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec4Swizzles;

    #[test]
    fn gpu_cpu_one_substep_single_triangle_positions_close() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let ctx = WgpuClothContext::new(&instance)
            .expect("GPU adapter + cloth_sim.wgsl load required for parity test");

        let mesh = triangle_cloth();
        assert_eq!(mesh.num_particles, 3);
        let n = mesh.num_particles;

        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;
        let mut u = ClothSimUniforms {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            num_particles: n,
            num_tris: (mesh.indices.len() / 3) as u32,
            gravity: Vec4::new(-1.0, -2.0, 0.5, 0.0),
            grab_idx: -1,
            grab_active: 0,
            grab_stiffness: 0.0,
            grab_target: Vec4::ZERO,
            ..Default::default()
        };
        u.inner_iterations = INNER_ITERS;
        u.constraint_batch_idx = 0;

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            grab_idx: u.grab_idx,
            grab_active: false,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        let mut sim_cpu = mesh.positions.clone();
        let mut prev = vec![Vec3::ZERO; n as usize];
        let mut vel = vec![Vec3::ZERO; n as usize];
        let mut jac_work = vec![Vec3::ZERO; n as usize];
        let rest3 = mesh.rest_positions.clone();

        xpbd_substep_with_self_collision(
            &mut sim_cpu,
            &mut prev,
            &mut vel,
            &mut jac_work,
            &mesh.inv_mass,
            &mesh.constraint_i,
            &mesh.constraint_j,
            &mesh.constraint_rest_len,
            &mesh.constraint_compliance,
            &rest3,
            u.thickness,
            u.coll_scale,
            &sub,
        );

        let b = make_buffers(&ctx, &mesh);
        ctx.queue.write_buffer(
            &b.params,
            0,
            bytemuck::bytes_of(&ClothSimParamsGpu::pack(&u, mesh.constraint_batch_count)),
        );
        let bg = make_bind_group(&ctx, &b);
        run_one_gpu_substep(&ctx, &b, &bg, &mesh, n, mesh.num_distance_constraints);
        let sim_gpu = read_vec4_positions(&ctx, &b.sim_pos, n as usize);

        let eps = 2e-3_f32;
        for i in 0..n as usize {
            let d = sim_gpu[i] - sim_cpu[i];
            let ad = d.x.abs().max(d.y.abs()).max(d.z.abs());
            assert!(
                ad < eps,
                "particle {i}: gpu {:?} cpu {:?} diff {:?}",
                sim_gpu[i],
                sim_cpu[i],
                d
            );
        }
    }

    #[test]
    fn gpu_cpu_grid_cloth_substep_positions_close() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let ctx = WgpuClothContext::new(&instance)
            .expect("GPU adapter + cloth_sim.wgsl load required for parity test");

        let mesh = crate::mesh_prep::grid_cloth_hanging(18, 18, 0.042);
        let n = mesh.num_particles;

        let sdt = REFERENCE_FRAME_DELTA_SECS / SUBSTEPS as f32;
        let mut u = ClothSimUniforms::default();
        u.dt = sdt;
        u.inv_dt = 1.0 / sdt;
        u.num_particles = n;
        u.num_tris = (mesh.indices.len() / 3) as u32;
        u.inner_iterations = INNER_ITERS;
        u.grab_idx = -1;
        u.grab_active = 0;
        u.constraint_batch_idx = 0;

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            grab_idx: u.grab_idx,
            grab_active: false,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        let mut sim_cpu = mesh.positions.clone();
        let mut prev = vec![Vec3::ZERO; n as usize];
        let mut vel = vec![Vec3::ZERO; n as usize];
        let mut jac_work = vec![Vec3::ZERO; n as usize];
        let rest3 = mesh.rest_positions.clone();

        xpbd_substep_with_self_collision(
            &mut sim_cpu,
            &mut prev,
            &mut vel,
            &mut jac_work,
            &mesh.inv_mass,
            &mesh.constraint_i,
            &mesh.constraint_j,
            &mesh.constraint_rest_len,
            &mesh.constraint_compliance,
            &rest3,
            u.thickness,
            u.coll_scale,
            &sub,
        );

        let b = make_buffers(&ctx, &mesh);
        ctx.queue.write_buffer(
            &b.params,
            0,
            bytemuck::bytes_of(&ClothSimParamsGpu::pack(&u, mesh.constraint_batch_count)),
        );
        let bg = make_bind_group(&ctx, &b);
        run_one_gpu_substep(&ctx, &b, &bg, &mesh, n, mesh.num_distance_constraints);
        let sim_gpu = read_vec4_positions(&ctx, &b.sim_pos, n as usize);

        let eps = 2.5e-2_f32;
        let mut max_err = 0.0_f32;
        for i in 0..n as usize {
            let d = sim_gpu[i] - sim_cpu[i];
            max_err = max_err.max(d.x.abs().max(d.y.abs()).max(d.z.abs()));
        }
        assert!(
            max_err < eps,
            "max |gpu-cpu| per component {} exceeds {} (atomics / ordering)",
            max_err,
            eps
        );
    }
}
