//! egui-driven debug overlay for the lumber camp.
//!
//! Top-right anchored window with FPS, per-segment CPU + GPU timings,
//! draw / instance / proxy counts, a rolling frametime chart, and a one-line
//! adapter footer. Stays out of the yakui game-UI's top-left status panel
//! and centre game-over banner. Wall-clock driven, so the trace keeps moving
//! when the sim is paused.

use std::collections::VecDeque;
use std::time::Duration;

use currawong::{CommandQueue, FrameStats, FrameTimings, SimEnvironment, egui, sun_direction_for};

use crate::sim::Command;

/// ~4 s of history at 60 fps. Same shape as `examples/trees.rs`.
pub const FRAMETIME_HISTORY: usize = 240;

/// Snapshot of what the overlay needs from the rest of the view.
pub struct DebugInput<'a> {
    pub samples: &'a VecDeque<f32>,
    pub timings: &'a FrameTimings,
    pub stats: &'a FrameStats,
    pub adapter_info: &'a str,
}

pub fn show(egui_ctx: &egui::Context, input: DebugInput<'_>) {
    let n = input.samples.len().max(1);
    let avg = input.samples.iter().sum::<f32>() / n as f32;
    let max = input.samples.iter().copied().fold(0.0f32, f32::max);
    let fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };

    egui::Window::new("debug")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .resizable(false)
        .collapsible(false)
        .show(egui_ctx, |ui| {
            ui.label(format!(
                "fps: {fps:5.1}   avg {:.2} ms   max {:.2} ms",
                avg * 1000.0,
                max * 1000.0,
            ));
            ui.add_space(2.0);

            // Per-segment timings. Monospaced so the columns align even as
            // the leading digits cross the 10-ms boundary.
            let gpu = input.timings.gpu;
            ui.monospace(format!("sim       {}", fmt_ms(Some(input.timings.sim_cpu))));
            ui.monospace(format!(
                "render    {}",
                fmt_ms(Some(input.timings.render_cpu))
            ));
            ui.monospace(format!("gpu world {}", fmt_ms(gpu.world)));
            ui.monospace(format!("gpu ui    {}", fmt_ms(gpu.overlay)));
            ui.monospace(format!("gpu total {}", fmt_ms(gpu.total())));

            // CPU counters.
            ui.add_space(2.0);
            let s = input.stats;
            ui.monospace(format!(
                "draws     {:>5}   inst {:>5}",
                s.draw_calls, s.instances
            ));
            ui.monospace(format!(
                "proxies   {:>5}   hyst {:>5}",
                s.proxies_visible, s.proxies_hysteresis
            ));

            draw_frametime_chart(ui, input.samples, max);

            // Adapter footer — small and dim so it stays out of the way.
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(input.adapter_info)
                    .small()
                    .color(egui::Color32::from_gray(160)),
            );
        });
}

/// Bottom-right anchored "environment" panel — time-of-day slider, day
/// length slider, day counter, sun-direction readout. Mutates sim state
/// through [`Command`] pushes (no `&mut Game` here, per the engine's
/// sim-mutation seam); the sim converts the float payloads to
/// [`SimUnit`](currawong::SimUnit) on apply.
pub fn show_environment(
    egui_ctx: &egui::Context,
    env: &SimEnvironment,
    cmds: &mut CommandQueue<Command>,
) {
    egui::Window::new("environment")
        .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
        .resizable(false)
        .collapsible(false)
        .show(egui_ctx, |ui| {
            // Day counter — incremented by `SimEnvironment::advance` when
            // `time_of_day` wraps past 1.0. Read-only; dragging the slider
            // past midnight on a single frame doesn't bump it (apply_command
            // assigns directly), which is the expected debug shape.
            ui.label(format!("day: {}", env.day));

            // Time-of-day slider. Local `f32` copy so the closure doesn't
            // need `&mut sim`; on user change we push a Command and the
            // sim picks it up at the next tick boundary.
            let mut time_of_day: f32 = env.time_of_day.to_num();
            let response = ui.add(
                egui::Slider::new(&mut time_of_day, 0.0..=1.0)
                    .text("time of day")
                    .custom_formatter(|v, _| format_clock_24h(v as f32)),
            );
            if response.changed() {
                cmds.push_now(Command::SetTimeOfDay(time_of_day));
            }

            // Day length — real-time seconds per in-world day. Logarithmic
            // so the slider has useful resolution across the 10s..600s
            // range (10s is "watch the sun race", 600s is the engine
            // default of one day per ten minutes).
            let mut seconds_per_day: f32 = env.seconds_per_day.to_num();
            let response = ui.add(
                egui::Slider::new(&mut seconds_per_day, 10.0..=600.0)
                    .text("day length (s)")
                    .logarithmic(true),
            );
            if response.changed() {
                cmds.push_now(Command::SetSecondsPerDay(seconds_per_day));
            }

            // Sun direction readout — the view's `extract_environment`
            // computes this same value and feeds the lit pass. Useful for
            // cross-checking the slider against the rendered lighting.
            let sun = sun_direction_for(env.time_of_day);
            ui.monospace(format!(
                "sun: ({:+.2}, {:+.2}, {:+.2})",
                sun.x, sun.y, sun.z
            ));
            ui.label(
                egui::RichText::new(if sun.z > 0.0 {
                    "above horizon"
                } else {
                    "below horizon"
                })
                .small()
                .color(egui::Color32::from_gray(160)),
            );
        });
}

/// `HH:MM` clock formatter for the time-of-day slider. Mirrors the helper
/// in `examples/textured_pbr.rs` so the two debug panels read alike.
fn format_clock_24h(t: f32) -> String {
    let total_minutes = (t.rem_euclid(1.0) * 24.0 * 60.0) as i32;
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    format!("{hours:02}:{minutes:02}")
}

/// Format a duration as `12.34 ms`, padded to a fixed width so the columns
/// in the breakdown stay aligned. `None` (GPU timing unsupported / not yet
/// readback) renders as a dash.
fn fmt_ms(d: Option<Duration>) -> String {
    match d {
        Some(d) => format!("{:>7.2} ms", d.as_secs_f32() * 1000.0),
        None => "      — ms".to_string(),
    }
}

/// Compact `"Backend: name (DeviceType)"` line for the footer. Built once
/// in `init` from [`Renderer::adapter_info`](currawong::Renderer::adapter_info).
pub fn format_adapter_info(info: &currawong::wgpu::AdapterInfo) -> String {
    format!(
        "{:?} · {} ({:?})",
        info.backend, info.name, info.device_type
    )
}

/// Rolling frametime strip chart drawn straight into the egui window via
/// `Painter` — no extra deps. The 16.6 ms (60 Hz) line is the budget;
/// samples poking above it are dropped frames.
fn draw_frametime_chart(ui: &mut egui::Ui, samples: &VecDeque<f32>, max_sample: f32) {
    const REF_DT: f32 = 1.0 / 60.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(240.0, 60.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(22));

    let y_max = (max_sample.max(REF_DT) * 1.2).max(1e-4);
    let y_for = |dt: f32| {
        let n = (dt / y_max).clamp(0.0, 1.0);
        rect.bottom() - n * rect.height()
    };

    let y_ref = y_for(REF_DT);
    painter.line_segment(
        [
            egui::pos2(rect.left(), y_ref),
            egui::pos2(rect.right(), y_ref),
        ],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 130, 80)),
    );

    if samples.len() >= 2 {
        let last_idx = (samples.len() - 1) as f32;
        let pts: Vec<egui::Pos2> = samples
            .iter()
            .enumerate()
            .map(|(i, dt)| {
                let x = rect.left() + (i as f32 / last_idx) * rect.width();
                egui::pos2(x, y_for(*dt))
            })
            .collect();
        painter.add(egui::Shape::line(
            pts,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(210, 220, 240)),
        ));
    }

    let label_font = egui::FontId::monospace(9.0);
    painter.text(
        rect.left_top() + egui::vec2(3.0, 1.0),
        egui::Align2::LEFT_TOP,
        format!("{:.1} ms", y_max * 1000.0),
        label_font.clone(),
        egui::Color32::from_gray(210),
    );
    painter.text(
        rect.left_bottom() + egui::vec2(3.0, -1.0),
        egui::Align2::LEFT_BOTTOM,
        "0 ms",
        label_font.clone(),
        egui::Color32::from_gray(140),
    );
    painter.text(
        egui::pos2(rect.left() + 3.0, y_ref + 1.0),
        egui::Align2::LEFT_TOP,
        "60 Hz",
        label_font,
        egui::Color32::from_rgb(150, 200, 150),
    );
}
