//! Extended PBR material: displaces vertices from GPU simulation `ShaderBuffer`s.

use bevy::{
    asset::embedded_asset,
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    reflect::TypePath,
    render::{render_resource::AsBindGroup, storage::ShaderBuffer},
    shader::ShaderRef,
};

const CLOTH_MAT_WGSL: &str = "embedded://bevy_softbody/shaders/cloth_vertex.wgsl";

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct ClothMatExt {
    #[storage(101, read_only)]
    pub sim_positions: Handle<ShaderBuffer>,
    #[storage(102, read_only)]
    pub sim_normals: Handle<ShaderBuffer>,
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
        embedded_asset!(app, "shaders/cloth_vertex.wgsl");
        app.add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, ClothMatExt>,
        >::default());
    }
}
