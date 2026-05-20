//! Example-local yakui widget conveniences.
//!
//! Composes the [`colored_box_container`] + [`nineslice`] + inner [`pad`]
//! sandwich into a single call so panel sites stay one indent deep instead
//! of three.

use currawong::yakui;
use yakui::widgets::{NineSlice, Pad};
use yakui::{Color, ManagedTextureId};

/// Window-style panel: solid background + nineslice frame + inner padding.
///
/// `texture` is the nineslice frame's [`ManagedTextureId`] (typically loaded
/// via [`currawong::YakuiAssets`]). `frame_margins` are the nineslice corner
/// pixels — match the artwork. `inner_pad` is breathing room between the
/// frame and the children. `bg` paints behind the frame (the frame's
/// transparent centre lets it show through).
pub fn panel(
    texture: ManagedTextureId,
    frame_margins: Pad,
    inner_pad: Pad,
    frame_tint: Color,
    bg: Color,
    children: impl FnOnce(),
) {
    yakui::colored_box_container(bg, || {
        NineSlice::new(texture, frame_margins, 1.0)
            .color(frame_tint)
            .show(|| {
                yakui::pad(inner_pad, children);
            });
    });
}
