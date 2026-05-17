//! [`Definitions`] — the pre-sim registry of namespaced "kinds".
//!
//! Loaded synchronously-from-the-sim's-perspective: by the time
//! `Simulation::new` runs, every `.ron` file under the configured prefix
//! has been read, parsed, and registered. The sim never reaches back into
//! the VFS for definitions.
//!
//! ## File shape
//!
//! Each `.ron` file is one kind. Top-level RON is a struct (or equivalent
//! map) with at least an `id` field naming the kind:
//!
//! ```ron
//! (
//!     id: "currawong:oak_tree",
//!     // ... arbitrary kind-specific fields, preserved as ron::Value
//! )
//! ```
//!
//! For PR1 the body is stored untyped as a [`ron::Value`]; typed kind
//! handlers will be wired up alongside the assets that need them (PR4).
//! Files outside the prefix or without a `.ron` extension are ignored.
//!
//! ## ID grammar
//!
//! `namespace:name` where both halves match `[a-z][a-z0-9_]*`. Same shape
//! as Factorio and Minecraft kind IDs and rejected for the same reasons
//! the issue cites: purely-numeric IDs are load-order-dependent under
//! modding (the engine can't tell `42` written by one mod from `42`
//! written by another after a load-order shuffle). Stable namespaced
//! strings sidestep that entirely.

use std::collections::HashMap;
use std::fmt;

use super::path::VfsPath;
use super::source::AssetError;
use super::vfs::Vfs;

/// A validated namespaced kind ID — `namespace:name`. Construct via
/// [`KindId::new`]; the constructor enforces the grammar so once you hold a
/// `KindId` it's structurally well-formed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KindId {
    full: String,
    /// Byte offset of `:` within `full`. Lets `namespace()` / `name()`
    /// return `&str` slices without re-scanning.
    sep: usize,
}

/// Reasons [`KindId::new`] rejects an input string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KindIdError {
    /// No `:` present, so we can't split into namespace + name.
    MissingSeparator,
    /// More than one `:` present.
    MultipleSeparators,
    /// Namespace half was empty.
    EmptyNamespace,
    /// Name half was empty.
    EmptyName,
    /// Namespace contained a disallowed character or didn't start with a
    /// lowercase letter.
    InvalidNamespace,
    /// Name contained a disallowed character or didn't start with a
    /// lowercase letter.
    InvalidName,
}

impl fmt::Display for KindIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSeparator => f.write_str("kind id is missing `:` separator"),
            Self::MultipleSeparators => f.write_str("kind id has more than one `:`"),
            Self::EmptyNamespace => f.write_str("kind id namespace is empty"),
            Self::EmptyName => f.write_str("kind id name is empty"),
            Self::InvalidNamespace => f.write_str("kind id namespace must match [a-z][a-z0-9_]*"),
            Self::InvalidName => f.write_str("kind id name must match [a-z][a-z0-9_]*"),
        }
    }
}

impl std::error::Error for KindIdError {}

impl KindId {
    pub fn new(input: impl Into<String>) -> Result<Self, KindIdError> {
        let input = input.into();
        let (ns, name) = input.split_once(':').ok_or(KindIdError::MissingSeparator)?;
        if name.contains(':') {
            return Err(KindIdError::MultipleSeparators);
        }
        if ns.is_empty() {
            return Err(KindIdError::EmptyNamespace);
        }
        if name.is_empty() {
            return Err(KindIdError::EmptyName);
        }
        if !is_valid_segment(ns) {
            return Err(KindIdError::InvalidNamespace);
        }
        if !is_valid_segment(name) {
            return Err(KindIdError::InvalidName);
        }
        let sep = ns.len();
        Ok(KindId { full: input, sep })
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

impl fmt::Display for KindId {
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

/// A parsed kind definition. The body is kept as the original
/// [`ron::Value`]; typed downcasts happen at use sites once the relevant
/// component types exist.
#[derive(Debug, Clone)]
pub struct KindDef {
    pub id: KindId,
    /// Where in the VFS this definition was loaded from. Useful for
    /// diagnostics ("oak_tree was defined in mods/foo/kinds/oak.ron").
    pub source: VfsPath,
    /// Untyped body, as parsed. Will be typed-deserialised by consumers
    /// that know the kind's component shape.
    pub value: ron::Value,
}

/// Errors that [`Definitions::load`] can return.
#[derive(Debug)]
pub enum DefError {
    /// Failed to read a definition file from the VFS.
    Read { path: VfsPath, error: AssetError },
    /// Failed to read the listing from the VFS.
    Listing(AssetError),
    /// File wasn't valid UTF-8 (RON is a text format; bytes-level mismatch
    /// should surface as a separate failure mode from a parse error so the
    /// user knows whether to fix the encoding or the syntax).
    NotUtf8 { path: VfsPath },
    /// RON parse failed.
    Parse {
        path: VfsPath,
        error: ron::de::SpannedError,
    },
    /// Top-level RON wasn't a struct/map, so there's nowhere to look for
    /// an `id` field.
    NotAStruct { path: VfsPath },
    /// Required `id` field was missing.
    MissingId { path: VfsPath },
    /// `id` field was present but wasn't a string.
    IdNotAString { path: VfsPath },
    /// `id` field was a string, but didn't parse as a [`KindId`].
    InvalidId { path: VfsPath, error: KindIdError },
    /// Two files registered the same `id`.
    Duplicate {
        id: KindId,
        first: VfsPath,
        second: VfsPath,
    },
}

impl fmt::Display for DefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, error } => write!(f, "reading {path}: {error}"),
            Self::Listing(e) => write!(f, "listing VFS: {e}"),
            Self::NotUtf8 { path } => write!(f, "{path}: file is not UTF-8"),
            Self::Parse { path, error } => write!(f, "parsing {path}: {error}"),
            Self::NotAStruct { path } => write!(f, "{path}: top-level RON must be a struct/map"),
            Self::MissingId { path } => write!(f, "{path}: missing `id` field"),
            Self::IdNotAString { path } => write!(f, "{path}: `id` field must be a string"),
            Self::InvalidId { path, error } => write!(f, "{path}: invalid kind id ({error})"),
            Self::Duplicate { id, first, second } => write!(
                f,
                "duplicate kind id `{id}`: defined in {first} and again in {second}",
            ),
        }
    }
}

impl std::error::Error for DefError {}

/// In-memory registry of all kind definitions loaded at startup.
#[derive(Debug, Default)]
pub struct Definitions {
    kinds: HashMap<KindId, KindDef>,
}

impl Definitions {
    /// Empty registry — useful for tests and for sims that don't use any
    /// definitions yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every `.ron` file under `prefix` from `vfs`, parse it, and
    /// register the resulting [`KindDef`].
    ///
    /// Hard-errors on the first malformed file rather than collecting and
    /// continuing. Tolerant loading (skip-and-warn) might be a useful mod
    /// loader behaviour later, but the engine should fail loud on its own
    /// content; defer the policy decision until a real mod harness exists.
    pub async fn load(vfs: &Vfs, prefix: &VfsPath) -> Result<Self, DefError> {
        let listing = vfs.list().map_err(DefError::Listing)?;
        let mut defs = Self::new();
        for path in listing {
            if !path.is_within(prefix) {
                continue;
            }
            if path.extension() != Some("ron") {
                continue;
            }
            let bytes = vfs.read(&path).await.map_err(|error| DefError::Read {
                path: path.clone(),
                error,
            })?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| DefError::NotUtf8 { path: path.clone() })?;
            let value: ron::Value = ron::from_str(text).map_err(|error| DefError::Parse {
                path: path.clone(),
                error,
            })?;
            let kind = parse_kind(path.clone(), value)?;
            if let Some(existing) = defs.kinds.get(&kind.id) {
                return Err(DefError::Duplicate {
                    id: kind.id.clone(),
                    first: existing.source.clone(),
                    second: kind.source.clone(),
                });
            }
            defs.kinds.insert(kind.id.clone(), kind);
        }
        Ok(defs)
    }

    pub fn get(&self, id: &KindId) -> Option<&KindDef> {
        self.kinds.get(id)
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&KindId, &KindDef)> {
        self.kinds.iter()
    }
}

/// Pull the `id` field out of a parsed RON value and build a [`KindDef`].
/// `ron::Map` doesn't expose `.get()`, so we iterate — fine on the load
/// path, and kind files have a tiny field count.
fn parse_kind(path: VfsPath, value: ron::Value) -> Result<KindDef, DefError> {
    let map = match &value {
        ron::Value::Map(m) => m,
        _ => return Err(DefError::NotAStruct { path }),
    };
    let id_value = map
        .iter()
        .find_map(|(k, v)| match k {
            ron::Value::String(s) if s == "id" => Some(v),
            _ => None,
        })
        .ok_or_else(|| DefError::MissingId { path: path.clone() })?;
    let id_str = match id_value {
        ron::Value::String(s) => s.clone(),
        _ => {
            return Err(DefError::IdNotAString { path: path.clone() });
        }
    };
    let id = KindId::new(id_str).map_err(|error| DefError::InvalidId {
        path: path.clone(),
        error,
    })?;
    Ok(KindDef {
        id,
        source: path,
        value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::memory_source::MemorySource;
    use crate::data::test_helpers::block_on;

    fn p(s: &str) -> VfsPath {
        VfsPath::new(s).unwrap()
    }

    fn vfs_with(files: &[(&str, &str)]) -> Vfs {
        let mut src = MemorySource::new();
        for (path, body) in files {
            src.insert(p(path), body.as_bytes().to_vec());
        }
        let mut vfs = Vfs::new();
        vfs.mount(src);
        vfs
    }

    // ----- KindId -----------------------------------------------------

    #[test]
    fn kind_id_happy() {
        let id = KindId::new("currawong:oak_tree").unwrap();
        assert_eq!(id.namespace(), "currawong");
        assert_eq!(id.name(), "oak_tree");
        assert_eq!(id.as_str(), "currawong:oak_tree");
    }

    #[test]
    fn kind_id_rejects() {
        assert_eq!(
            KindId::new("no_separator").unwrap_err(),
            KindIdError::MissingSeparator
        );
        assert_eq!(
            KindId::new("a:b:c").unwrap_err(),
            KindIdError::MultipleSeparators
        );
        assert_eq!(
            KindId::new(":name").unwrap_err(),
            KindIdError::EmptyNamespace
        );
        assert_eq!(KindId::new("ns:").unwrap_err(), KindIdError::EmptyName);
        assert_eq!(
            KindId::new("1ns:name").unwrap_err(),
            KindIdError::InvalidNamespace
        );
        assert_eq!(
            KindId::new("NS:name").unwrap_err(),
            KindIdError::InvalidNamespace
        );
        assert_eq!(
            KindId::new("ns:1name").unwrap_err(),
            KindIdError::InvalidName
        );
        assert_eq!(
            KindId::new("ns:Name").unwrap_err(),
            KindIdError::InvalidName
        );
        assert_eq!(
            KindId::new("ns:name-with-dash").unwrap_err(),
            KindIdError::InvalidName
        );
    }

    // ----- Definitions loading ---------------------------------------

    #[test]
    fn loads_one_kind() {
        let vfs = vfs_with(&[(
            "kinds/oak.ron",
            r#"(id: "currawong:oak_tree", max_height: 12.0)"#,
        )]);
        let defs = block_on(Definitions::load(&vfs, &p("kinds"))).unwrap();
        assert_eq!(defs.len(), 1);
        let id = KindId::new("currawong:oak_tree").unwrap();
        let def = defs.get(&id).expect("oak registered");
        assert_eq!(def.id, id);
        assert_eq!(def.source.as_str(), "kinds/oak.ron");
    }

    #[test]
    fn ignores_files_outside_prefix() {
        let vfs = vfs_with(&[
            ("kinds/oak.ron", r#"(id: "currawong:oak_tree")"#),
            (
                "other/skip.ron",
                r#"(id: "currawong:should_not_be_loaded")"#,
            ),
        ]);
        let defs = block_on(Definitions::load(&vfs, &p("kinds"))).unwrap();
        assert_eq!(defs.len(), 1);
        assert!(
            defs.get(&KindId::new("currawong:oak_tree").unwrap())
                .is_some()
        );
        assert!(
            defs.get(&KindId::new("currawong:should_not_be_loaded").unwrap())
                .is_none()
        );
    }

    #[test]
    fn ignores_non_ron_files() {
        let vfs = vfs_with(&[
            ("kinds/oak.ron", r#"(id: "currawong:oak_tree")"#),
            ("kinds/readme.txt", "this should not be parsed as RON"),
        ]);
        let defs = block_on(Definitions::load(&vfs, &p("kinds"))).unwrap();
        assert_eq!(defs.len(), 1);
    }

    #[test]
    fn loads_multiple_kinds() {
        let vfs = vfs_with(&[
            ("kinds/oak.ron", r#"(id: "currawong:oak_tree")"#),
            ("kinds/birch.ron", r#"(id: "currawong:birch_tree")"#),
            ("kinds/sub/pine.ron", r#"(id: "currawong:pine_tree")"#),
        ]);
        let defs = block_on(Definitions::load(&vfs, &p("kinds"))).unwrap();
        assert_eq!(defs.len(), 3);
    }

    #[test]
    fn malformed_ron_surfaces_parse_error() {
        let vfs = vfs_with(&[("kinds/oak.ron", r#"(id: "currawong:oak_tree""#)]);
        match block_on(Definitions::load(&vfs, &p("kinds"))) {
            Err(DefError::Parse { path, .. }) => assert_eq!(path.as_str(), "kinds/oak.ron"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn missing_id_surfaces_clear_error() {
        let vfs = vfs_with(&[("kinds/oak.ron", r#"(max_height: 12.0)"#)]);
        match block_on(Definitions::load(&vfs, &p("kinds"))) {
            Err(DefError::MissingId { path }) => assert_eq!(path.as_str(), "kinds/oak.ron"),
            other => panic!("expected MissingId, got {other:?}"),
        }
    }

    #[test]
    fn invalid_id_surfaces_clear_error() {
        let vfs = vfs_with(&[("kinds/oak.ron", r#"(id: "no_colon_here")"#)]);
        match block_on(Definitions::load(&vfs, &p("kinds"))) {
            Err(DefError::InvalidId { path, error }) => {
                assert_eq!(path.as_str(), "kinds/oak.ron");
                assert_eq!(error, KindIdError::MissingSeparator);
            }
            other => panic!("expected InvalidId, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_id_surfaces_clear_error() {
        // Both files in the same layer, parsed in listing order — we can't
        // predict which comes first, but Duplicate should always fire.
        let vfs = vfs_with(&[
            ("kinds/a.ron", r#"(id: "currawong:oak_tree")"#),
            ("kinds/b.ron", r#"(id: "currawong:oak_tree")"#),
        ]);
        match block_on(Definitions::load(&vfs, &p("kinds"))) {
            Err(DefError::Duplicate { id, first, second }) => {
                assert_eq!(id.as_str(), "currawong:oak_tree");
                assert_ne!(first, second);
            }
            other => panic!("expected Duplicate, got {other:?}"),
        }
    }

    #[test]
    fn id_not_a_string_surfaces_clear_error() {
        let vfs = vfs_with(&[("kinds/oak.ron", r#"(id: 42)"#)]);
        match block_on(Definitions::load(&vfs, &p("kinds"))) {
            Err(DefError::IdNotAString { path }) => assert_eq!(path.as_str(), "kinds/oak.ron"),
            other => panic!("expected IdNotAString, got {other:?}"),
        }
    }
}
