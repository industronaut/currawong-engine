//! Sim-side state for the editor: a single zone with one object at the
//! origin, plus the [`Command`] enum the View pushes for kind selection,
//! hot reload, and in-memory bounds edits.
//!
//! `Game::new` parses every kind's `render`, `interaction`, and `footprint`
//! blocks once at startup, caches the typed values, and seeds the subject's
//! [`KindId`] component to the first available kind. From there the only
//! way sim state changes is through [`Game::apply_command`] — matching the
//! Command-as-only-mutation invariant from the root CLAUDE.md.

use std::collections::HashMap;
use std::time::Duration;

use currawong::data::{Definitions, KindId};
use currawong::{
    Facing, Footprint, Interaction, RenderSpec, SimPos, SimUnit, Simulation, WorldObjectId,
    WorldTransform, Zone, ZoneId, Zones,
};

/// Sim-side mutations: kind selection from the egui panel, plus the
/// hot-reload reload-typed-digest pushed by the View when the file watcher
/// detects a change under `assets/`.
#[derive(Debug, Clone)]
pub(crate) enum Command {
    SelectKind(KindId),
    /// Replace the sim's parsed-at-startup caches with a freshly-loaded
    /// snapshot. The view parses [`Definitions`] off the file-watcher tick
    /// (it's a tiny directory; cheap) and ships the typed digest through
    /// here so [`apply_command`](Game::apply_command) stays a synchronous
    /// in-memory swap.
    ReloadDefinitions {
        available: Vec<KindId>,
        render_specs: HashMap<KindId, RenderSpec>,
        interactions: HashMap<KindId, Interaction>,
        footprints: HashMap<KindId, Footprint>,
    },
    /// Replace the `bounds_min`/`bounds_max` of a kind's cached
    /// [`RenderSpec`] in memory. Drives the recalc-from-mesh button: the
    /// edit lives in sim state and is reflected by the bounds overlay +
    /// camera auto-frame on the next frame. The Save button mirrors the
    /// in-memory value out to the kind's `.ron` file separately.
    UpdateBounds {
        kind: KindId,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    },
}

pub(crate) struct Game {
    pub(crate) zones: Zones,
    pub(crate) zone: ZoneId,
    pub(crate) subject: WorldObjectId,
    /// Sorted list of every kind that has a `render` block — the source for
    /// the egui kind list. Sim-side because the UI reads it via `&Sim`.
    pub(crate) available: Vec<KindId>,
    /// Cached `RenderSpec` per kind for camera auto-framing. Cheap (a few
    /// dozen entries max in any reasonable kinds folder); avoids re-parsing
    /// the def on every selection.
    pub(crate) render_specs: HashMap<KindId, RenderSpec>,
    /// Cached `Interaction` per kind for the interaction-tiles overlay.
    /// Same shape as `render_specs` — sim-side typed parse of the def, done
    /// once at startup so the view can `.get(&kind).tiles(transform)` each
    /// frame without re-parsing. Kinds whose def omits `interaction:`
    /// deserialize to [`Interaction::None`], which `tiles` resolves to an
    /// empty vec — the overlay simply draws zero instances.
    pub(crate) interactions: HashMap<KindId, Interaction>,
    /// Cached `Footprint` per kind for the placement-tiles overlay. Same
    /// once-at-startup parse pattern as `interactions`; kinds without a
    /// `tiles:` field deserialize to an empty `Footprint` and draw zero
    /// instances.
    pub(crate) footprints: HashMap<KindId, Footprint>,
}

impl Game {
    pub(crate) fn new(defs: Definitions) -> Self {
        let mut available: Vec<KindId> = Vec::new();
        let mut render_specs: HashMap<KindId, RenderSpec> = HashMap::new();
        let mut interactions: HashMap<KindId, Interaction> = HashMap::new();
        let mut footprints: HashMap<KindId, Footprint> = HashMap::new();
        for (kind_id, def) in defs.iter() {
            match RenderSpec::from_def(def) {
                Ok(spec) => {
                    available.push(kind_id.clone());
                    render_specs.insert(kind_id.clone(), spec);
                    // Interaction is independent of render: a kind without a
                    // declared interaction still appears in the editor, just
                    // with an empty tile overlay. A parse error here would
                    // mean a malformed `interaction:` block — eprintln but
                    // fall through to `Interaction::None` so the kind is
                    // still selectable for visual inspection.
                    let interaction = Interaction::from_def(def).unwrap_or_else(|e| {
                        eprintln!("lumber_editor: {kind_id} interaction parse: {e}");
                        Interaction::None
                    });
                    interactions.insert(kind_id.clone(), interaction);
                    // Footprint follows the same opt-in convention; missing
                    // `tiles:` defaults to an empty footprint.
                    let footprint = Footprint::from_def(def).unwrap_or_else(|e| {
                        eprintln!("lumber_editor: {kind_id} footprint parse: {e}");
                        Footprint::default()
                    });
                    footprints.insert(kind_id.clone(), footprint);
                }
                Err(e) => {
                    eprintln!("lumber_editor: skipping {kind_id}: {e}");
                }
            }
        }
        available.sort_by(|a, b| a.as_str().cmp(b.as_str()));

        let mut zones = Zones::new();
        let zone_id = zones.insert(Zone::new());
        let zone = zones.get_mut(zone_id).expect("just inserted");
        let subject = zone.insert(WorldTransform {
            position: SimPos::new(SimUnit::ZERO, SimUnit::ZERO, SimUnit::ZERO),
            facing: Facing::ZERO,
        });
        if let Some(first) = available.first() {
            zone.components_mut().insert(subject, first.clone());
        }

        Self {
            zones,
            zone: zone_id,
            subject,
            available,
            render_specs,
            interactions,
            footprints,
        }
    }
}

impl Simulation for Game {
    type Command = Command;

    fn tick(&mut self, _dt: Duration) {}

    fn apply_command(&mut self, cmd: &Command) {
        match cmd {
            Command::SelectKind(kind) => {
                if let Some(zone) = self.zones.get_mut(self.zone) {
                    zone.components_mut().insert(self.subject, kind.clone());
                }
            }
            Command::ReloadDefinitions {
                available,
                render_specs,
                interactions,
                footprints,
            } => {
                self.available = available.clone();
                self.render_specs = render_specs.clone();
                self.interactions = interactions.clone();
                self.footprints = footprints.clone();
                // If the previously-selected kind disappeared from defs
                // (renamed, deleted, or its `render:` block went away),
                // fall back to the first available so the editor doesn't
                // get stuck pointing at a phantom.
                if let Some(zone) = self.zones.get_mut(self.zone) {
                    let current = zone.components().get::<KindId>(self.subject).cloned();
                    let still_valid = current.as_ref().is_some_and(|k| self.available.contains(k));
                    if !still_valid && let Some(first) = self.available.first() {
                        zone.components_mut().insert(self.subject, first.clone());
                    }
                }
            }
            Command::UpdateBounds { kind, min, max } => {
                if let Some(spec) = self.render_specs.get_mut(kind) {
                    spec.bounds_min = *min;
                    spec.bounds_max = *max;
                }
            }
        }
    }
}
