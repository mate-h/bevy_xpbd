//! egui panel for the cloth example.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass};
use bevy_softbody::cloth_compute::{
    ClothSimConfig, ClothSimControl, ClothSimUniforms, DEFAULT_COLL_SCALE,
    DEFAULT_LINEAR_AIR_DRAG_PER_SEC, THICKNESS,
};
#[cfg(feature = "solver-jacobi")]
use bevy_softbody::cloth_jacobi::jacobi_default_omega;

/// Rolling frame-time graph length (~2 s at 60 Hz).
const FRAME_GRAPH_SAMPLES: usize = 120;
/// Target 60 Hz reference line on the graph (ms).
const FRAME_GRAPH_REF_MS: f32 = 1000.0 / 60.0;

/// Ring buffer of recent wall-clock frame durations (seconds).
#[derive(Resource, Default)]
struct FrameTimeHistory {
    samples: Vec<f32>,
    head: usize,
}

impl FrameTimeHistory {
    fn push(&mut self, dt_secs: f32) {
        if self.samples.len() < FRAME_GRAPH_SAMPLES {
            self.samples.push(dt_secs);
            return;
        }
        self.samples[self.head] = dt_secs;
        self.head = (self.head + 1) % FRAME_GRAPH_SAMPLES;
    }

    fn chronological(&self) -> Vec<f32> {
        let n = self.samples.len();
        if n < FRAME_GRAPH_SAMPLES {
            return self.samples.clone();
        }
        let mut out = Vec::with_capacity(FRAME_GRAPH_SAMPLES);
        out.extend_from_slice(&self.samples[self.head..]);
        out.extend_from_slice(&self.samples[..self.head]);
        out
    }
}

pub struct ClothUiPlugin;

impl Plugin for ClothUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .insert_resource(EguiGlobalSettings {
                enable_absorb_bevy_input_system: true,
                ..default()
            })
            .init_resource::<FrameTimeHistory>()
            .add_systems(Update, record_frame_time)
            .add_systems(EguiPrimaryContextPass, sim_params_ui);
    }
}

fn record_frame_time(time: Res<Time>, mut history: ResMut<FrameTimeHistory>) {
    history.push(time.delta_secs());
}

fn frame_time_graph(ui: &mut egui::Ui, history: &FrameTimeHistory) {
    let samples = history.chronological();
    let height = 52.0;
    let width = ui.available_width();
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(28, 30, 36));

    if samples.is_empty() {
        return;
    }

    let latest_ms = samples.last().copied().unwrap_or(0.0) * 1000.0;
    let peak_ms = samples
        .iter()
        .map(|dt| dt * 1000.0)
        .fold(FRAME_GRAPH_REF_MS, f32::max);
    let ymax_ms = peak_ms.max(FRAME_GRAPH_REF_MS * 1.5).min(100.0);

    let y_for_ms = |ms: f32| rect.bottom() - (ms / ymax_ms).clamp(0.0, 1.0) * rect.height();
    let ref_y = y_for_ms(FRAME_GRAPH_REF_MS);
    painter.line_segment(
        [
            egui::pos2(rect.left(), ref_y),
            egui::pos2(rect.right(), ref_y),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 75, 85)),
    );

    let stroke = egui::Stroke::new(
        1.5,
        if latest_ms <= FRAME_GRAPH_REF_MS * 1.1 {
            egui::Color32::from_rgb(90, 200, 120)
        } else if latest_ms <= FRAME_GRAPH_REF_MS * 2.0 {
            egui::Color32::from_rgb(230, 190, 70)
        } else {
            egui::Color32::from_rgb(230, 90, 90)
        },
    );

    let n = samples.len();
    for (i, window) in samples.windows(2).enumerate() {
        let x0 = rect.left() + (i as f32 / (n - 1).max(1) as f32) * rect.width();
        let x1 = rect.left() + ((i + 1) as f32 / (n - 1).max(1) as f32) * rect.width();
        let y0 = y_for_ms(window[0] * 1000.0);
        let y1 = y_for_ms(window[1] * 1000.0);
        painter.line_segment([egui::pos2(x0, y0), egui::pos2(x1, y1)], stroke);
    }

    if samples.len() == 1 {
        let y = y_for_ms(samples[0] * 1000.0);
        painter.circle_filled(egui::pos2(rect.right(), y), 2.0, stroke.color);
    }

    let fps = if latest_ms > 0.0 {
        1000.0 / latest_ms
    } else {
        0.0
    };
    ui.label(format!("Frame: {latest_ms:.1} ms  ({fps:.0} fps)"));
}

fn sim_params_ui(
    mut contexts: EguiContexts,
    mut config: ResMut<ClothSimConfig>,
    mut uniforms: ResMut<ClothSimUniforms>,
    mut ctrl: ResMut<ClothSimControl>,
    history: Res<FrameTimeHistory>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("Simulation")
        .default_pos([12.0, 12.0])
        .default_width(280.0)
        .show(ctx, |ui| {
            let solver_label = if cfg!(feature = "solver-jacobi") {
                "Jacobi"
            } else {
                "Gauss–Seidel"
            };
            ui.label(format!("Solver: {solver_label}"));
            ui.label(format!("Substep dt: {:.4} ms", uniforms.dt * 1000.0));
            ui.label("Frame time");
            frame_time_graph(ui, &history);

            ui.separator();
            ui.heading("Solver");
            ui.add(egui::Slider::new(&mut config.solve_substeps, 1..=64).text("substeps / frame"));
            ui.add(
                egui::Slider::new(&mut config.solve_inner_iterations, 1..=48)
                    .text("inner iterations"),
            );
            #[cfg(feature = "solver-jacobi")]
            ui.add(egui::Slider::new(&mut uniforms.jacobi_omega, 0.05..=1.0).text("Jacobi ω"));
            ui.add(
                egui::Slider::new(&mut config.collision_every_n_substeps, 1..=16)
                    .text("collision every N substeps"),
            );

            ui.separator();
            ui.heading("Forces");
            let mut gravity_y = uniforms.gravity.y;
            if ui
                .add(egui::Slider::new(&mut gravity_y, -30.0..=0.0).text("gravity Y"))
                .changed()
            {
                uniforms.gravity.y = gravity_y;
            }
            ui.add(
                egui::Slider::new(&mut uniforms.linear_drag_per_sec, 0.0..=8.0)
                    .text("air drag (1/s)"),
            );

            ui.separator();
            ui.heading("Collisions");
            ui.add(egui::Slider::new(&mut uniforms.thickness, 0.005..=0.12).text("thickness"));
            ui.add(egui::Slider::new(&mut uniforms.coll_scale, 0.0..=1.0).text("self-collision"));

            ui.separator();
            ui.heading("Grab");
            ui.add(egui::Slider::new(&mut uniforms.grab_stiffness, 0.05..=1.0).text("stiffness"));

            ui.separator();
            ui.heading("Playback");
            ui.checkbox(&mut ctrl.sim_paused, "Paused (P)");
            if ui.button("Step frame (N)").clicked() {
                ctrl.step_serial = ctrl.step_serial.saturating_add(1);
            }
            if ui.button("Reinit simulation").clicked() {
                ctrl.reinit_serial = ctrl.reinit_serial.saturating_add(1);
            }

            ui.separator();
            if ui.button("Reset defaults").clicked() {
                reset_sim_defaults(&mut config, &mut uniforms);
            }
        });
    Ok(())
}

fn reset_sim_defaults(config: &mut ClothSimConfig, uniforms: &mut ClothSimUniforms) {
    config.solve_substeps = 24;
    #[cfg(feature = "solver-jacobi")]
    {
        config.solve_inner_iterations = 20;
    }
    #[cfg(feature = "solver-gauss-seidel")]
    {
        config.solve_inner_iterations = 8;
    }
    config.collision_every_n_substeps = 4;
    uniforms.thickness = THICKNESS;
    uniforms.coll_scale = DEFAULT_COLL_SCALE;
    uniforms.gravity.y = -9.81;
    uniforms.linear_drag_per_sec = DEFAULT_LINEAR_AIR_DRAG_PER_SEC;
    uniforms.grab_stiffness = 0.25;
    #[cfg(feature = "solver-jacobi")]
    {
        uniforms.jacobi_omega = jacobi_default_omega();
    }
}
