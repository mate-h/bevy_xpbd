//! XPBD cloth example: GPU simulation + extended PBR + mouse grab.
//!
//! Loads welded `assets/cloth.obj` (crate root) via `bevy_xpbd::mesh_prep::parse_welded_obj`.
//! If the file is missing or empty, falls back to a procedural grid.

use std::path::PathBuf;

use bevy::{color::LinearRgba, input::keyboard::KeyCode, input::mouse::MouseButton, pbr::ExtendedMaterial, prelude::*};
use bevy::pbr::StandardMaterial;
use bevy::render::storage::ShaderStorageBuffer;
use bevy_xpbd::{
    cloth_compute::{
        ClothComputePlugin, ClothSimConfig, ClothSimControl, ClothSimUniforms, DEFAULT_COLL_SCALE,
        THICKNESS,
    },
    cloth_material::{ClothMaterialPlugin, ClothMatExt},
    mesh_prep::{grid_cloth_hanging, parse_welded_obj, ClothMeshData},
};

/// Relative to the crate root (`CARGO_MANIFEST_DIR`).
const CLOTH_OBJ_PATH: &str = "assets/cloth.obj";

/// Procedural fallback — if sim jitters or stretches, restore defaults via `solve_substeps: 36`,
/// `solve_inner_iterations: 22` (`bevy_xpbd::cloth_compute`) or widen `cell_size` / simpler mesh.
const CLOTH_QUAD_COLS: u32 = 24;
const CLOTH_QUAD_ROWS: u32 = 18;
const CLOTH_CELL_SIZE: f32 = 0.045;

fn load_cloth_mesh() -> ClothMeshData {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CLOTH_OBJ_PATH);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let data = parse_welded_obj(&text);
            if data.num_particles == 0 {
                warn!(
                    "{} parsed to zero particles; using procedural grid",
                    path.display()
                );
                grid_cloth_hanging(CLOTH_QUAD_COLS, CLOTH_QUAD_ROWS, CLOTH_CELL_SIZE)
            } else {
                info!(
                    "Loaded cloth from {} ({} particles, {} tris)",
                    path.display(),
                    data.num_particles,
                    data.indices.len() / 3
                );
                data
            }
        }
        Err(e) => {
            warn!(
                "Could not read {}: {e}; using procedural grid",
                path.display()
            );
            grid_cloth_hanging(CLOTH_QUAD_COLS, CLOTH_QUAD_ROWS, CLOTH_CELL_SIZE)
        }
    }
}

#[derive(Resource, Default)]
struct GrabState {
    active: bool,
    particle: i32,
    ray_t: f32,
}

/// World-space rest positions (static) for picking in local space via inverse transform.
#[derive(Resource)]
struct ClothPickMesh {
    local_rest: Vec<Vec3>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy XPBD cloth — P pause  N step  (sim starts paused)".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((ClothMaterialPlugin, ClothComputePlugin))
        .insert_resource(ClothSimUniforms::default())
        .insert_resource(GrabState::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (mouse_grab, cloth_sim_debug_keys))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, ClothMatExt>>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut uniforms: ResMut<ClothSimUniforms>,
) {
    let cloth = load_cloth_mesh();
    let config = cloth.to_sim_config(&mut *buffers);

    {
        let u = uniforms.as_mut();
        u.num_particles = config.num_particles;
        u.num_tris = config.num_tris;
        u.thickness = THICKNESS;
        u.coll_scale = DEFAULT_COLL_SCALE;
    }

    commands.insert_resource(ClothSimConfig {
        // ~40% fewer total dispatches vs (36×22) at typical `B`; raise both if seams stretch.
        solve_substeps: 24,
        solve_inner_iterations: 14,
        // Halves pairwise self-collision work when `coll_scale > 0` (runs on odd-indexed substeps).
        collision_every_n_substeps: 2,
        render_positions: config.render_positions.clone(),
        render_normals: config.render_normals.clone(),
        ..config
    });

    commands.insert_resource(ClothPickMesh {
        local_rest: cloth.positions.clone(),
    });

    let mesh = cloth.to_bevy_mesh();
    let mat = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: LinearRgba::new(0.08, 0.28, 0.92, 1.0).into(),
            double_sided: true,
            ..default()
        },
        extension: ClothMatExt {
            sim_positions: config.render_positions,
            sim_normals: config.render_normals,
        },
    });

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(mat),
        Transform::from_xyz(0.0, 0.35, 0.0).with_scale(Vec3::splat(1.0)),
    ));

    commands.spawn((
        DirectionalLight::default(),
        Transform::from_translation(Vec3::new(4.0, 8.0, 2.0))
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.15, 2.25).looking_at(Vec3::new(0.0, 0.1, 0.0), Vec3::Y),
    ));
}

fn cloth_sim_debug_keys(mut ctrl: ResMut<ClothSimControl>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyP) {
        ctrl.sim_paused = !ctrl.sim_paused;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        ctrl.step_serial = ctrl.step_serial.saturating_add(1);
    }
}

fn mouse_grab(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    cloth_xf: Query<&GlobalTransform, With<Mesh3d>>,
    pick: Res<ClothPickMesh>,
    mut grab: ResMut<GrabState>,
    mut uniforms: ResMut<ClothSimUniforms>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let Some(cursor) = win.cursor_position() else {
        return;
    };
    let Ok((cam, cam_gtf)) = camera_q.single() else {
        return;
    };
    let Ok(cloth_gtf) = cloth_xf.single() else {
        return;
    };

    let Ok(ray) = cam.viewport_to_world(cam_gtf, cursor) else {
        return;
    };
    let ro = ray.origin;
    let rd = ray.direction.normalize();

    let inv = cloth_gtf.affine().inverse();
    let lo = inv.transform_point3(ro);
    let ld = inv.transform_vector3(rd).normalize();

    if buttons.just_pressed(MouseButton::Left) {
        if let Some((idx, t_hit)) = pick_closest_vertex(lo, ld, &pick.local_rest) {
            grab.active = true;
            grab.particle = idx as i32;
            grab.ray_t = t_hit;
            uniforms.grab_idx = idx as i32;
            uniforms.grab_active = 1;
        }
    }

    if buttons.just_released(MouseButton::Left) {
        grab.active = false;
        grab.particle = -1;
        uniforms.grab_idx = -1;
        uniforms.grab_active = 0;
    }

    if grab.active {
        uniforms.grab_target = (lo + ld * grab.ray_t).extend(0.0);
    }
}

fn pick_closest_vertex(origin: Vec3, dir: Vec3, pts: &[Vec3]) -> Option<(usize, f32)> {
    let d2_line = |p: Vec3| {
        let t = (p - origin).dot(dir) / dir.dot(dir).max(1e-9);
        (t, (origin + dir * t - p).length_squared())
    };
    let mut best_i = None;
    let mut best_t = 0.0_f32;
    let mut best_d2 = f32::MAX;
    for (i, p) in pts.iter().enumerate() {
        let (t, err2) = d2_line(*p);
        if t < 0.0 {
            continue;
        }
        if err2 < best_d2 {
            best_d2 = err2;
            best_i = Some(i);
            best_t = t;
        }
    }
    best_i.map(|i| (i, best_t))
}
