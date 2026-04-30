#import bevy_pbr::{
    mesh_functions,
    mesh_bindings::mesh,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    pbr_deferred_functions::deferred_output,
}
#endif

#import bevy_pbr::forward_io::FragmentOutput

@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> sim_positions: array<vec4<f32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<storage, read> sim_normals: array<vec4<f32>>;

@vertex
fn vertex(vertex: Vertex, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
#ifdef MORPH_TARGETS
    return out;
#endif
    // Particle id: prefer UV1 — matches rest mesh (`uv_b.x = particle index`) when `VERTEX_UVS_B` is on.
    // Fallback: indexed draw identity (`vertex_index` + Bevy slab base), same as `draw_indexed` base vertex.
#ifdef VERTEX_UVS_B
    let vi = u32(vertex.uv_b.x + 0.5);
#else
    let vi = vertex_index - mesh[vertex.instance_index].first_vertex_index;
#endif

    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
#ifdef SKINNED
    var world_from_local = bevy_pbr::skinning::skin_model(
        vertex.joint_indices,
        vertex.joint_weights,
        vertex.instance_index
    );
#else
    var world_from_local = mesh_world_from_local;
#endif

#ifdef VERTEX_NORMALS
#ifdef SKINNED
    out.world_normal = bevy_pbr::skinning::skin_normals(world_from_local, vertex.normal);
#else
    let sn = sim_normals[vi].xyz;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(sn, vertex.instance_index);
#endif
#endif

#ifdef VERTEX_POSITIONS
    let sp = sim_positions[vi].xyz;
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(sp, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex.instance_index,
    );
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex.instance_index,
        mesh_world_from_local[3],
    );
#endif

    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
