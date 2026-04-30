//! Extended PBR material: displaces vertices from GPU simulation `ShaderStorageBuffer`s.

use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::AsBindGroup,
        storage::{ShaderStorageBuffer},
    },
    shader::ShaderRef,
};

const CLOTH_MAT_WGSL: &str = "shaders/cloth_vertex.wgsl";

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct ClothMatExt {
    #[storage(101, read_only)]
    pub sim_positions: Handle<ShaderStorageBuffer>,
    #[storage(102, read_only)]
    pub sim_normals: Handle<ShaderStorageBuffer>,
}

impl MaterialExtension for ClothMatExt {
    fn enable_prepass() -> bool {
        false
    }

    fn vertex_shader() -> ShaderRef {
        CLOTH_MAT_WGSL.into()
    }

    fn fragment_shader() -> ShaderRef {
        CLOTH_MAT_WGSL.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        CLOTH_MAT_WGSL.into()
    }
}

pub struct ClothMaterialPlugin;

impl Plugin for ClothMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, ClothMatExt>>::default());
    }
}
