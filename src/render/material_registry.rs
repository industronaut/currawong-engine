//! Name-keyed registry of material instances, paired with the
//! source-prefixed [`MaterialId`] used to look them up.
//!
//! Mirrors [`RenderRegistry`](super::RenderRegistry): keyed by a stable
//! namespaced string ID that's authored once and referenced from elsewhere
//! (kind defs, glb material slots, save files). Where the render registry
//! resolves *templates* from a `KindId`, this one resolves *material
//! instances* from a [`MaterialId`].
//!
//! ## Why a separate registry from [`MaterialInstanceRegistry`](super::MaterialInstanceRegistry)
//!
//! The existing [`MaterialInstanceRegistry<I, K>`](super::MaterialInstanceRegistry)
//! is keyed by a user-chosen enum (`enum Mat { Wood, Stone }`) — fine when
//! the call site enumerates its materials at compile time. The shape #80
//! needs is different: glb files name their material slots as strings
//! (`currawong:mat_bark`), and the loader has to look those up against
//! whatever the application registered without knowing the user's enum.
//! Stable string keys solve that the way [`KindId`](crate::data::KindId)
//! solves it for sim kinds.
//!
//! ## ID grammar
//!
//! [`MaterialId`] enforces the same `namespace:name` shape as
//! [`KindId`](crate::data::KindId) — both halves `[a-z][a-z0-9_]*`, exactly
//! one `:`. Stylistic prefixes like `mat_` / `mesh_` inside the name half
//! are a *convention*, not a grammar rule; the type doesn't know about
//! them. Conventionally:
//!
//! ```text
//! currawong:mat_bark        # engine-shipped material
//! mymod:mat_bark            # mod-defined material that overlays the engine one
//! ```
//!
//! ## Fallback on miss
//!
//! Lookup is best-effort: [`MaterialRegistry::get_by_name`] takes a raw
//! string (typically straight off a glb primitive's material slot),
//! attempts to parse it as a [`MaterialId`], and returns `None` if either
//! the parse fails or no instance is registered. Callers compose this with
//! the magenta fallback already designed in #67 — a primitive with an
//! unregistered material name draws through a magenta-flavoured material
//! instance the application chose to register as the fallback.
//!
//! Bare glb material names with no `:` (the default when authors don't
//! rename Blender's material slots — `"Lumber"`, `"Wood.001"`) are
//! defaulted to the `gltf:` namespace and lowercased, so registering
//! `gltf:lumber` is enough to catch a slot named `"Lumber"` straight out
//! of Blender. Authors who want to overlay or override an engine material
//! still spell the full `currawong:mat_foo` form inside the .blend.

use std::collections::HashMap;
use std::fmt;

/// A validated namespaced material ID — `namespace:name`. Same grammar as
/// [`KindId`](crate::data::KindId); kept as a distinct newtype so the type
/// system distinguishes sim-side kind identities from view-side material
/// identities at API boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialId {
    full: String,
    /// Byte offset of `:` within `full`. Lets `namespace()` / `name()`
    /// return `&str` slices without re-scanning.
    sep: usize,
}

/// Reasons [`MaterialId::new`] rejects an input string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterialIdError {
    MissingSeparator,
    MultipleSeparators,
    EmptyNamespace,
    EmptyName,
    InvalidNamespace,
    InvalidName,
}

impl fmt::Display for MaterialIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSeparator => f.write_str("material id is missing `:` separator"),
            Self::MultipleSeparators => f.write_str("material id has more than one `:`"),
            Self::EmptyNamespace => f.write_str("material id namespace is empty"),
            Self::EmptyName => f.write_str("material id name is empty"),
            Self::InvalidNamespace => {
                f.write_str("material id namespace must match [a-z][a-z0-9_]*")
            }
            Self::InvalidName => f.write_str("material id name must match [a-z][a-z0-9_]*"),
        }
    }
}

impl std::error::Error for MaterialIdError {}

impl MaterialId {
    pub fn new(input: impl Into<String>) -> Result<Self, MaterialIdError> {
        let input = input.into();
        let (ns, name) = input
            .split_once(':')
            .ok_or(MaterialIdError::MissingSeparator)?;
        if name.contains(':') {
            return Err(MaterialIdError::MultipleSeparators);
        }
        if ns.is_empty() {
            return Err(MaterialIdError::EmptyNamespace);
        }
        if name.is_empty() {
            return Err(MaterialIdError::EmptyName);
        }
        if !is_valid_segment(ns) {
            return Err(MaterialIdError::InvalidNamespace);
        }
        if !is_valid_segment(name) {
            return Err(MaterialIdError::InvalidName);
        }
        let sep = ns.len();
        Ok(MaterialId { full: input, sep })
    }

    pub fn as_str(&self) -> &str {
        &self.full
    }

    pub fn namespace(&self) -> &str {
        &self.full[..self.sep]
    }

    pub fn name(&self) -> &str {
        &self.full[self.sep + 1..]
    }
}

impl fmt::Display for MaterialId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.full)
    }
}

fn is_valid_segment(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Registry of material instances keyed by [`MaterialId`].
///
/// Generic over `I`, the concrete instance type the caller registers
/// (commonly [`PbrMaterialInstance`](super::PbrMaterialInstance)). Build
/// once at `View::init`; reuse the same registry across every glb the
/// application loads. Re-registering an id silently replaces the previous
/// instance — same idiom as
/// [`RenderRegistry::register`](super::RenderRegistry::register).
///
/// Lookups come in two flavours:
/// - [`get`](Self::get) takes a parsed [`MaterialId`] and is the fast path
///   for application code that already holds typed identities.
/// - [`get_by_name`](Self::get_by_name) takes a raw `&str` (typically a glb
///   primitive's material slot name), attempts to parse it, and returns
///   `None` on either a parse miss or a registry miss. Callers fall back
///   to a magenta-flavoured instance on `None`.
pub struct MaterialRegistry<I> {
    instances: HashMap<MaterialId, I>,
}

impl<I> MaterialRegistry<I> {
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }

    /// Register `instance` under `id`. Replaces any existing entry with
    /// the same id; the old instance is dropped.
    pub fn register(&mut self, id: MaterialId, instance: I) {
        self.instances.insert(id, instance);
    }

    /// Look up the instance registered under `id`. Returns `None` if no
    /// instance was registered.
    pub fn get(&self, id: &MaterialId) -> Option<&I> {
        self.instances.get(id)
    }

    /// Look up the instance for a raw glb-style material slot name.
    /// Returns `None` if either the name fails [`MaterialId`] validation
    /// or no instance is registered under the parsed id.
    ///
    /// Bare names with no `:` (typical of unmodified Blender material
    /// slots — `"Lumber"`, `"Wood"`) are defaulted to the `gltf:`
    /// namespace and lowercased, so `"Lumber"` resolves to `gltf:lumber`.
    /// Authors who want a different namespace prefix their material name
    /// in Blender (e.g. `currawong:mat_oak`) and the explicit form passes
    /// through unchanged.
    pub fn get_by_name(&self, raw: &str) -> Option<&I> {
        let normalized;
        let id_input: &str = if raw.contains(':') {
            raw
        } else {
            normalized = format!("gltf:{}", raw.to_lowercase());
            &normalized
        };
        let id = MaterialId::new(id_input).ok()?;
        self.instances.get(&id)
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Iterate every `(id, instance)` pair the registry holds. Order is
    /// unspecified (`HashMap`-backed).
    pub fn iter(&self) -> impl Iterator<Item = (&MaterialId, &I)> {
        self.instances.iter()
    }

    /// Mutable counterpart to [`iter`](Self::iter). Reserved for use
    /// cases that need to mutate the stored instances in place; today the
    /// per-frame [`MeshMaterial::refresh`](super::MeshMaterial) pass
    /// takes `&self` (interior mutability inside each instance) and
    /// walks through [`iter`](Self::iter) instead.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&MaterialId, &mut I)> {
        self.instances.iter_mut()
    }
}

impl<I> Default for MaterialRegistry<I> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_id_round_trips() {
        let id = MaterialId::new("currawong:mat_bark").expect("valid id");
        assert_eq!(id.as_str(), "currawong:mat_bark");
        assert_eq!(id.namespace(), "currawong");
        assert_eq!(id.name(), "mat_bark");
        assert_eq!(format!("{id}"), "currawong:mat_bark");
    }

    #[test]
    fn missing_separator_rejected() {
        let err = MaterialId::new("Lumber").expect_err("must reject unnamespaced name");
        assert_eq!(err, MaterialIdError::MissingSeparator);
    }

    #[test]
    fn multiple_separators_rejected() {
        assert_eq!(
            MaterialId::new("a:b:c").unwrap_err(),
            MaterialIdError::MultipleSeparators,
        );
    }

    #[test]
    fn empty_halves_rejected() {
        assert_eq!(
            MaterialId::new(":name").unwrap_err(),
            MaterialIdError::EmptyNamespace,
        );
        assert_eq!(
            MaterialId::new("ns:").unwrap_err(),
            MaterialIdError::EmptyName,
        );
    }

    #[test]
    fn invalid_chars_rejected() {
        // Uppercase isn't in [a-z][a-z0-9_]*.
        assert_eq!(
            MaterialId::new("Currawong:mat_bark").unwrap_err(),
            MaterialIdError::InvalidNamespace,
        );
        assert_eq!(
            MaterialId::new("currawong:Mat_Bark").unwrap_err(),
            MaterialIdError::InvalidName,
        );
        // Leading digit isn't allowed.
        assert_eq!(
            MaterialId::new("currawong:1mat").unwrap_err(),
            MaterialIdError::InvalidName,
        );
    }

    #[test]
    fn empty_registry_misses_everything() {
        let reg: MaterialRegistry<u32> = MaterialRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        let id = MaterialId::new("currawong:mat_bark").unwrap();
        assert!(reg.get(&id).is_none());
        assert!(reg.get_by_name("currawong:mat_bark").is_none());
        assert!(reg.get_by_name("Lumber").is_none());
    }

    #[test]
    fn register_then_lookup() {
        let mut reg = MaterialRegistry::new();
        let bark = MaterialId::new("currawong:mat_bark").unwrap();
        let stone = MaterialId::new("currawong:mat_stone").unwrap();
        reg.register(bark.clone(), 1u32);
        reg.register(stone.clone(), 2u32);

        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get(&bark), Some(&1));
        assert_eq!(reg.get(&stone), Some(&2));
        assert_eq!(reg.get_by_name("currawong:mat_bark"), Some(&1));
    }

    #[test]
    fn re_register_replaces() {
        let mut reg = MaterialRegistry::new();
        let id = MaterialId::new("currawong:mat_bark").unwrap();
        reg.register(id.clone(), 1u32);
        reg.register(id.clone(), 42u32);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get(&id), Some(&42));
    }

    #[test]
    fn get_by_name_misses_on_parse_failure() {
        let mut reg = MaterialRegistry::new();
        let id = MaterialId::new("currawong:mat_bark").unwrap();
        reg.register(id, 1u32);
        // "Lumber" is defaulted to "gltf:lumber" — still a miss because
        // nothing's registered under that id. Explicit `currawong:` lookup
        // for an unregistered name also misses.
        assert!(reg.get_by_name("Lumber").is_none());
        assert!(reg.get_by_name("currawong:mat_unregistered").is_none());
    }

    #[test]
    fn get_by_name_defaults_bare_to_gltf_namespace() {
        let mut reg = MaterialRegistry::new();
        let id = MaterialId::new("gltf:lumber").unwrap();
        reg.register(id, 42u32);
        // Blender's default slot name lands here lowercased + namespaced.
        assert_eq!(reg.get_by_name("Lumber"), Some(&42));
        assert_eq!(reg.get_by_name("LUMBER"), Some(&42));
        assert_eq!(reg.get_by_name("lumber"), Some(&42));
        // Explicit form passes through unchanged.
        assert_eq!(reg.get_by_name("gltf:lumber"), Some(&42));
        // Authors who hand-spell a different namespace get exact-match.
        assert!(reg.get_by_name("currawong:lumber").is_none());
    }

    #[test]
    fn get_by_name_bare_with_invalid_chars_misses() {
        let mut reg = MaterialRegistry::new();
        let id = MaterialId::new("gltf:lumber").unwrap();
        reg.register(id, 1u32);
        // Defaulting lowercases but doesn't munge the rest — a space or
        // leading digit still fails grammar validation.
        assert!(reg.get_by_name("Lumber Camp").is_none());
        assert!(reg.get_by_name("3wood").is_none());
    }
}
