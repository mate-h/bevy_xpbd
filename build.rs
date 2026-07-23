use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut features = Vec::new();
    if std::env::var_os("CARGO_FEATURE_SOLVER_JACOBI").is_some() {
        features.push("solver-jacobi".into());
    }
    if std::env::var_os("CARGO_FEATURE_SOLVER_GAUSS_SEIDEL").is_some() {
        features.push("solver-gauss-seidel".into());
    }
    if features.is_empty() {
        features.push("solver-jacobi".into());
    }

    let mut builder = SpirvBuilder::new("crates/softbody_solver", "spirv-unknown-vulkan1.1");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);
    builder
        .shader_crate_default_features(false)
        .shader_crate_features(features)
        // Match host `bytemuck` / WGSL packing (scalar/vec2 at non-16 offsets).
        .uniform_buffer_standard_layout(true)
        .build()?;
    Ok(())
}
