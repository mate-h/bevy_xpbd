//! Compare one GPU substep (raw `wgpu` dispatch of `cloth_sim.wgsl`) to [`crate::xpbd_cpu`].

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::path::Path;

use bevy::math::{Vec3, Vec4};

use crate::cloth_compute::{
    ClothSimParamsGpu, ClothSimUniforms, DT, INNER_ITERS, SUBSTEPS,
};
use crate::mesh_prep::ClothMeshData;
use crate::xpbd_cpu::{xpbd_substep_with_self_collision, XpbdCpuTimeStepParams};

fn neighbor_packed(mesh: &ClothMeshData) -> Vec<Vec4> {
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

fn vec4_buf(n: usize) -> u64 {
    (n * 16) as u64
}

struct WgpuClothContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    predict: wgpu::ComputePipeline,
    copy_sim_to_jac: wgpu::ComputePipeline,
    copy_jac_to_sim: wgpu::ComputePipeline,
    jacobi_edges: wgpu::ComputePipeline,
    jacobi_gather: wgpu::ComputePipeline,
    clear_constraint_lambda: wgpu::ComputePipeline,
    post_velocity: wgpu::ComputePipeline,
    clear_atomics: wgpu::ComputePipeline,
    collide_pairs: wgpu::ComputePipeline,
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

        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders/cloth_sim.wgsl");
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
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
                storage_entry(9, true),
                storage_entry(10, true),
                storage_entry(11, true),
                storage_entry(12, true),
                storage_entry(13, true),
                storage_entry(14, false),
                storage_entry(15, false),
                storage_entry(16, true),
                storage_entry(17, false),
                storage_entry(18, false),
                storage_entry(19, false),
                storage_entry(20, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloth"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        macro_rules! cp {
            ($entry:literal) => {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some($entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some($entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
        }

        Some(Self {
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

struct Buffers {
    params: wgpu::Buffer,
    sim_pos: wgpu::Buffer,
    jac_a: wgpu::Buffer,
    jac_b: wgpu::Buffer,
    prev: wgpu::Buffer,
    vel: wgpu::Buffer,
    rest: wgpu::Buffer,
    inv_mass: wgpu::Buffer,
    neigh_off: wgpu::Buffer,
    neigh_pack: wgpu::Buffer,
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
}

fn make_buffers(ctx: &WgpuClothContext, mesh: &ClothMeshData, packed: &[Vec4]) -> Buffers {
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

    let sim_pos = vb("sim", vec4_buf(n));
    let jac_a = vb("jac_a", vec4_buf(n));
    let jac_b = vb("jac_b", vec4_buf(n));
    let prev = vb("prev", vec4_buf(n));
    let vel = vb("vel", vec4_buf(n));
    let rest = vb("rest", vec4_buf(n));
    let render_pos = vb("rp", vec4_buf(n));
    let render_nrm = vb("rn", vec4_buf(n));

    let ip = bytemuck::cast_slice::<Vec4, u8>(&initial_pos);
    ctx.queue.write_buffer(&sim_pos, 0, ip);
    ctx.queue.write_buffer(&jac_a, 0, ip);
    ctx.queue.write_buffer(&prev, 0, ip);
    ctx.queue.write_buffer(&rest, 0, bytemuck::cast_slice::<Vec4, u8>(&rest_pos));
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
        label: Some("inv_mass"),
        size: (n * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &inv_mass,
        0,
        bytemuck::cast_slice::<f32, u8>(&mesh.inv_mass),
    );

    let neigh_off = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neigh_off"),
        size: ((n + 1) * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &neigh_off,
        0,
        bytemuck::cast_slice::<u32, u8>(&mesh.neighbor_offsets),
    );

    let neigh_pack = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neigh_pack"),
        size: (packed.len() * 16) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&neigh_pack, 0, bytemuck::cast_slice::<Vec4, u8>(packed));

    let ec = mesh.num_distance_constraints as usize;
    let ec_store = ec.max(1);
    let constraint_i = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_i"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_j = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_j"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_rest = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_rest"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_comp = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_comp"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_lambda = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_lambda"),
        size: (ec_store * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    let constraint_delta_lambda = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("constraint_delta_lambda"),
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
    ctx.queue.write_buffer(
        &constraint_lambda,
        0,
        &vec![0u8; ec_store * 4],
    );
    ctx.queue.write_buffer(
        &constraint_delta_lambda,
        0,
        &vec![0u8; ec_store * 4],
    );

    let tri = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tri"),
        size: (mesh.indices.len() * 4) as u64,
        usage,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(
        &tri,
        0,
        bytemuck::cast_slice::<u32, u8>(&mesh.indices),
    );

    let n3 = (n * 3 * 4) as u64;
    let atomic_coll = vb("atom_coll", n3);
    let atomic_norm = vb("atom_norm", n3);

    let params = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<ClothSimParamsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Buffers {
        params,
        sim_pos,
        jac_a,
        jac_b,
        prev,
        vel,
        rest,
        inv_mass,
        neigh_off,
        neigh_pack,
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
    }
}

fn make_bind_group(
    ctx: &WgpuClothContext,
    b: &Buffers,
    jac_in: &wgpu::Buffer,
    jac_out: &wgpu::Buffer,
) -> wgpu::BindGroup {
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
                resource: jac_in.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: jac_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b.prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: b.vel.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: b.rest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: b.inv_mass.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: b.neigh_off.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: b.neigh_pack.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: b.constraint_i.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: b.constraint_j.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: b.constraint_rest.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: b.constraint_comp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: b.constraint_lambda.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: b.constraint_delta_lambda.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: b.tri.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: b.render_pos.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 18,
                resource: b.render_nrm.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 19,
                resource: b.atomic_coll.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 20,
                resource: b.atomic_norm.as_entire_binding(),
            },
        ],
    })
}

struct BindTriplet {
    base: wgpu::BindGroup,
    jac_b_to_a: wgpu::BindGroup,
    jac_a_to_b: wgpu::BindGroup,
    copy_from_b: wgpu::BindGroup,
}

fn bind_triplet(ctx: &WgpuClothContext, b: &Buffers) -> BindTriplet {
    BindTriplet {
        base: make_bind_group(ctx, b, &b.jac_a, &b.jac_b),
        jac_b_to_a: make_bind_group(ctx, b, &b.jac_b, &b.jac_a),
        jac_a_to_b: make_bind_group(ctx, b, &b.jac_a, &b.jac_b),
        copy_from_b: make_bind_group(ctx, b, &b.jac_b, &b.jac_a),
    }
}

fn run_one_gpu_substep(ctx: &WgpuClothContext, _b: &Buffers, bg: &BindTriplet, n: u32, num_constraints: u32) {
    let wg64 = ((n as usize) + 63) / 64;
    let wg256 = (n as usize * 3 + 255) / 256;
    let pairs = n as usize * ((n as usize).saturating_sub(1)) / 2;
    let wg_pairs = (pairs + 255) / 256;
    let wg_constraints = (num_constraints as usize + 63) / 64;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloth_substep"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("substep"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.predict);
        pass.set_bind_group(0, &bg.base, &[]);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);

        pass.set_pipeline(&ctx.copy_sim_to_jac);
        pass.set_bind_group(0, &bg.base, &[]);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);

        if num_constraints > 0 {
            for k in 0..INNER_ITERS {
                pass.set_pipeline(&ctx.clear_constraint_lambda);
                pass.set_bind_group(0, &bg.base, &[]);
                pass.dispatch_workgroups(wg_constraints as u32, 1, 1);

                let jac_bg = if k % 2 == 0 {
                    &bg.jac_b_to_a
                } else {
                    &bg.jac_a_to_b
                };
                pass.set_pipeline(&ctx.jacobi_edges);
                pass.set_bind_group(0, jac_bg, &[]);
                pass.dispatch_workgroups(wg_constraints as u32, 1, 1);

                pass.set_pipeline(&ctx.jacobi_gather);
                pass.set_bind_group(0, jac_bg, &[]);
                pass.dispatch_workgroups(wg64 as u32, 1, 1);
            }
        }

        pass.set_pipeline(&ctx.copy_jac_to_sim);
        pass.set_bind_group(0, &bg.copy_from_b, &[]);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);

        pass.set_pipeline(&ctx.clear_atomics);
        pass.set_bind_group(0, &bg.base, &[]);
        pass.dispatch_workgroups(wg256 as u32, 1, 1);

        pass.set_pipeline(&ctx.collide_pairs);
        pass.set_bind_group(0, &bg.base, &[]);
        pass.dispatch_workgroups(wg_pairs as u32, 1, 1);

        pass.set_pipeline(&ctx.collide_apply);
        pass.set_bind_group(0, &bg.base, &[]);
        pass.dispatch_workgroups(wg64 as u32, 1, 1);

        pass.set_pipeline(&ctx.post_velocity);
        pass.set_bind_group(0, &bg.base, &[]);
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
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_sim"),
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
    flat.into_iter()
        .map(|v| Vec3::new(v.x, v.y, v.z))
        .collect()
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let ctx = WgpuClothContext::new(&instance)
            .expect("GPU adapter + cloth_sim.wgsl load required for parity test");

        let mesh = triangle_cloth();
        assert_eq!(mesh.num_particles, 3);
        let packed = neighbor_packed(&mesh);
        let n = mesh.num_particles;

        let sdt = DT / SUBSTEPS as f32;
        let mut u = ClothSimUniforms {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            num_particles: n,
            num_tris: (mesh.indices.len() / 3) as u32,
            gravity: Vec4::new(-1.0, -2.0, 0.5, 0.0),
            floor_y: -10.0,
            grab_idx: -1,
            grab_active: 0,
            grab_stiffness: 0.0,
            grab_target: Vec4::ZERO,
            ..Default::default()
        };
        u.inner_iterations = INNER_ITERS;

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: u.grab_idx,
            grab_active: false,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        let mut sim_cpu = mesh.positions.clone();
        let mut prev = vec![Vec3::ZERO; n as usize];
        let mut vel = vec![Vec3::ZERO; n as usize];
        let mut jac_a = vec![Vec3::ZERO; n as usize];
        let mut jac_b = vec![Vec3::ZERO; n as usize];
        let rest3 = mesh.rest_positions.clone();

        xpbd_substep_with_self_collision(
            &mut sim_cpu,
            &mut prev,
            &mut vel,
            &mut jac_a,
            &mut jac_b,
            &mesh.inv_mass,
            &mesh.neighbor_offsets,
            &mesh.neighbor_other,
            &mesh.neighbor_constraint_id,
            &mesh.constraint_i,
            &mesh.constraint_j,
            &mesh.constraint_rest_len,
            &mesh.constraint_compliance,
            &rest3,
            u.thickness,
            u.coll_scale,
            &sub,
        );

        let b = make_buffers(&ctx, &mesh, &packed);
        ctx.queue.write_buffer(
            &b.params,
            0,
            bytemuck::bytes_of(&ClothSimParamsGpu::from(&u)),
        );
        let bg = bind_triplet(&ctx, &b);
        run_one_gpu_substep(&ctx, &b, &bg, n, mesh.num_distance_constraints);
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let ctx = WgpuClothContext::new(&instance)
            .expect("GPU adapter + cloth_sim.wgsl load required for parity test");

        let mesh = crate::mesh_prep::grid_cloth_hanging(18, 18, 0.042);
        let packed = neighbor_packed(&mesh);
        let n = mesh.num_particles;

        let sdt = DT / SUBSTEPS as f32;
        let mut u = ClothSimUniforms::default();
        u.dt = sdt;
        u.inv_dt = 1.0 / sdt;
        u.num_particles = n;
        u.num_tris = (mesh.indices.len() / 3) as u32;
        u.inner_iterations = INNER_ITERS;
        u.grab_idx = -1;
        u.grab_active = 0;

        let sub = XpbdCpuTimeStepParams {
            dt: sdt,
            inv_dt: 1.0 / sdt,
            jacobi_omega: u.jacobi_omega,
            inner_iterations: INNER_ITERS,
            gravity: u.gravity.xyz(),
            floor_y: u.floor_y,
            grab_idx: u.grab_idx,
            grab_active: false,
            grab_target: u.grab_target.xyz(),
            grab_stiffness: u.grab_stiffness,
            linear_drag_per_sec: u.linear_drag_per_sec,
        };

        let mut sim_cpu = mesh.positions.clone();
        let mut prev = vec![Vec3::ZERO; n as usize];
        let mut vel = vec![Vec3::ZERO; n as usize];
        let mut jac_a = vec![Vec3::ZERO; n as usize];
        let mut jac_b = vec![Vec3::ZERO; n as usize];
        let rest3 = mesh.rest_positions.clone();

        xpbd_substep_with_self_collision(
            &mut sim_cpu,
            &mut prev,
            &mut vel,
            &mut jac_a,
            &mut jac_b,
            &mesh.inv_mass,
            &mesh.neighbor_offsets,
            &mesh.neighbor_other,
            &mesh.neighbor_constraint_id,
            &mesh.constraint_i,
            &mesh.constraint_j,
            &mesh.constraint_rest_len,
            &mesh.constraint_compliance,
            &rest3,
            u.thickness,
            u.coll_scale,
            &sub,
        );

        let b = make_buffers(&ctx, &mesh, &packed);
        ctx.queue.write_buffer(
            &b.params,
            0,
            bytemuck::bytes_of(&ClothSimParamsGpu::from(&u)),
        );
        let bg = bind_triplet(&ctx, &b);
        run_one_gpu_substep(&ctx, &b, &bg, n, mesh.num_distance_constraints);
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