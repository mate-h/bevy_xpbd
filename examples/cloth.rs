//! XPBD cloth example: GPU simulation + extended PBR + mouse grab.
//!
//! Default build uses **parallel Jacobi** (`solver-jacobi`). For **colored Gauss–Seidel**:
//! `cargo run --example cloth_xpbd --no-default-features --features solver-gauss-seidel`.

#[path = "cloth/cloth_ui.rs"]
mod cloth_ui;

use bevy::pbr::StandardMaterial;
use bevy::render::storage::ShaderBuffer;
use bevy::{
    color::LinearRgba, input::keyboard::KeyCode, input::mouse::MouseButton, pbr::ExtendedMaterial,
    prelude::*,
};
use bevy_egui::input::{egui_wants_any_keyboard_input, EguiWantsInput};
use bevy_softbody::{
    cloth_compute::{
        ClothComputePlugin, ClothSimConfig, ClothSimControl, ClothSimFrameTiming, ClothSimUniforms,
        DEFAULT_COLL_SCALE, THICKNESS,
    },
    cloth_material::{ClothMatExt, ClothMaterialPlugin},
    mesh_prep::{grid_cloth_hanging, ClothMeshData},
};
use cloth_ui::ClothUiPlugin;

/// Low-pass blend for wall-clock frame Δt (`ClothSimFrameTiming::blend_alpha`).
const FRAME_DELTA_BLEND_ALPHA: f32 = 0.05;

const CLOTH_QUAD_COLS: u32 = 128;
const CLOTH_QUAD_ROWS: u32 = 96;
const CLOTH_CELL_SIZE: f32 = (24.0 * 0.1 * 2.0) / CLOTH_QUAD_COLS as f32;

fn procedural_cloth() -> ClothMeshData {
    grid_cloth_hanging(CLOTH_QUAD_COLS, CLOTH_QUAD_ROWS, CLOTH_CELL_SIZE)
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
    inv_mass: Vec<f32>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy XPBD cloth".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((ClothMaterialPlugin, ClothComputePlugin, ClothUiPlugin))
        .insert_resource(ClothSimUniforms::default())
        .insert_resource(GrabState::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                on_cloth_reinit,
                mouse_grab,
                cloth_sim_debug_keys.run_if(not(egui_wants_any_keyboard_input)),
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, ClothMatExt>>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut uniforms: ResMut<ClothSimUniforms>,
    mut timing: ResMut<ClothSimFrameTiming>,
) {
    timing.blend_alpha = FRAME_DELTA_BLEND_ALPHA;

    let cloth = procedural_cloth();
    let config = cloth.to_sim_config(&mut *buffers);

    {
        let u = uniforms.as_mut();
        u.num_particles = config.num_particles;
        u.num_tris = config.num_tris;
        u.thickness = THICKNESS;
        // Collision is the other big cost; keep mild + sparse for the demo.
        u.coll_scale = DEFAULT_COLL_SCALE * 0.35;
        // Kinematic grab: stiffness is follow rate toward the cursor (not a spring vs XPBD).
        u.grab_stiffness = 0.45;
    }

    commands.insert_resource(ClothSimConfig {
        // Realtime budget: kinematic grab stays stable without huge iteration counts.
        solve_substeps: 10,
        #[cfg(feature = "solver-jacobi")]
        solve_inner_iterations: 6,
        #[cfg(feature = "solver-gauss-seidel")]
        solve_inner_iterations: 4,
        collision_every_n_substeps: 8,
        render_positions: config.render_positions.clone(),
        render_normals: config.render_normals.clone(),
        ..config
    });

    commands.insert_resource(ClothPickMesh {
        local_rest: cloth.positions.clone(),
        inv_mass: cloth.inv_mass.clone(),
    });

    let mesh = cloth.to_bevy_mesh();
    let mat = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: LinearRgba::new(0.08, 0.28, 0.92, 1.0).into(),
            // `double_sided` only flips normals in the shader; back faces still need `cull_mode: None`
            // so the rasterizer draws them at all (`StandardMaterial` defaults to back-face culling).
            double_sided: true,
            cull_mode: None,
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
        Transform::from_xyz(0.0, -0.65, 0.0).with_scale(Vec3::splat(1.0)),
    ));

    commands.spawn((
        DirectionalLight::default(),
        Transform::from_translation(Vec3::new(4.0, 8.0, 2.0))
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

fn on_cloth_reinit(
    ctrl: Res<ClothSimControl>,
    mut last_reinit: Local<u64>,
    mut grab: ResMut<GrabState>,
    mut uniforms: ResMut<ClothSimUniforms>,
) {
    if ctrl.reinit_serial == *last_reinit {
        return;
    }
    *last_reinit = ctrl.reinit_serial;
    if ctrl.reinit_serial == 0 {
        return;
    }
    release_grab(&mut grab, &mut uniforms);
}

fn cloth_sim_debug_keys(mut ctrl: ResMut<ClothSimControl>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyP) {
        ctrl.sim_paused = !ctrl.sim_paused;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        ctrl.step_serial = ctrl.step_serial.saturating_add(1);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        ctrl.reinit_serial = ctrl.reinit_serial.saturating_add(1);
    }
}

fn release_grab(grab: &mut GrabState, uniforms: &mut ClothSimUniforms) {
    grab.active = false;
    grab.particle = -1;
    uniforms.grab_idx = -1;
    uniforms.grab_active = 0;
}

fn mouse_grab(
    egui_wants: Res<EguiWantsInput>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    cloth_xf: Query<&GlobalTransform, With<Mesh3d>>,
    pick: Res<ClothPickMesh>,
    mut grab: ResMut<GrabState>,
    mut uniforms: ResMut<ClothSimUniforms>,
) {
    if egui_wants.wants_any_pointer_input() {
        if grab.active {
            release_grab(&mut grab, &mut uniforms);
        }
        return;
    }

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
        if let Some((idx, t_hit)) = pick_closest_vertex(lo, ld, &pick.local_rest, &pick.inv_mass) {
            grab.active = true;
            grab.particle = idx as i32;
            // Depth of the rest particle along the view ray (plane facing the camera).
            // Mouse motion then slides on that plane — avoids deep/near jumps that crumple corners.
            grab.ray_t = t_hit.max(0.05);
            uniforms.grab_idx = idx as i32;
            uniforms.grab_active = 1;
            uniforms.grab_target = (lo + ld * grab.ray_t).extend(0.0);
        }
    }

    if buttons.just_released(MouseButton::Left) {
        release_grab(&mut grab, &mut uniforms);
    }

    if grab.active {
        // Keep a fixed camera-depth plane from press time; only slide laterally with the cursor.
        uniforms.grab_target = (lo + ld * grab.ray_t).extend(0.0);
    }
}

fn pick_closest_vertex(
    origin: Vec3,
    dir: Vec3,
    pts: &[Vec3],
    inv_mass: &[f32],
) -> Option<(usize, f32)> {
    let d2_line = |p: Vec3| {
        let t = (p - origin).dot(dir) / dir.dot(dir).max(1e-9);
        (t, (origin + dir * t - p).length_squared())
    };
    let mut best_i = None;
    let mut best_t = 0.0_f32;
    let mut best_d2 = f32::MAX;
    for (i, p) in pts.iter().enumerate() {
        if inv_mass.get(i).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
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
