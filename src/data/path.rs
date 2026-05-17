//! [`VfsPath`] — the type-level grammar for paths inside the VFS.
//!
//! The constructor is the only way to build one, and it rejects every form
//! that could let an asset reference escape the sandbox: parent traversal
//! (`..`), absolute paths (leading `/`), drive letters / alternate data
//! streams (any `:`), backslashes, embedded NULs, and `.` no-op segments.
//! The string is then normalised (repeated slashes collapsed, trailing
//! slash stripped) so equal paths hash equal.
//!
//! Once built, a [`VfsPath`] is a plain string under the hood — it's
//! suitable as a `HashMap` key, save-file reference, and mod-override
//! target. Lookups never have to re-normalise.

use std::fmt;

/// A normalised, sandbox-safe path inside the virtual filesystem.
///
/// Construct via [`VfsPath::new`]; the constructor enforces the path
/// grammar (see [`VfsPathError`] for the rejected forms). Forward-slash
/// only — host-OS conventions (Windows backslashes, drive letters) are
/// rejected, not silently translated. This is deliberate: the VFS is a
/// flat namespace, not a view onto the host filesystem.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VfsPath(String);

/// Reasons [`VfsPath::new`] rejects an input string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsPathError {
    /// Input was empty or contained only slashes.
    Empty,
    /// Input started with `/` (absolute paths are not allowed; the VFS has
    /// no notion of root other than the empty prefix).
    Absolute,
    /// Input contained a backslash (`\`). VFS paths are forward-slash only;
    /// we don't translate Windows-style paths to keep the contract simple.
    Backslash,
    /// Input contained a colon (`:`). This catches Windows drive letters
    /// (`C:/foo`) and NTFS alternate data streams (`file:stream`); the
    /// colon is also reserved as the separator in [`KindId`](crate::data::KindId)
    /// so keeping VFS paths colon-free avoids confusion when a path is
    /// embedded in a kind ID context.
    Colon,
    /// Input contained an embedded NUL byte. POSIX uses NUL as the C-string
    /// terminator, so any path containing one is a path-injection vector.
    NulByte,
    /// Input contained a `..` segment. Parent traversal would let an asset
    /// reference escape the VFS sandbox (e.g. a malicious mod reading
    /// `../../etc/passwd`); rejecting `..` at the type level makes that
    /// structurally impossible.
    ParentTraversal,
    /// Input contained a `.` segment. Allowing `.` would mean `a/./b` and
    /// `a/b` are different strings but the same path; rejecting it keeps
    /// equality cheap (a string compare) rather than requiring callers to
    /// re-normalise on every comparison.
    CurrentDir,
}

impl fmt::Display for VfsPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("path is empty"),
            Self::Absolute => f.write_str("path is absolute (starts with `/`)"),
            Self::Backslash => f.write_str("path contains a backslash"),
            Self::Colon => f.write_str("path contains a colon"),
            Self::NulByte => f.write_str("path contains a NUL byte"),
            Self::ParentTraversal => f.write_str("path contains a `..` segment"),
            Self::CurrentDir => f.write_str("path contains a `.` segment"),
        }
    }
}

impl std::error::Error for VfsPathError {}

impl VfsPath {
    /// Validate and normalise `input` into a [`VfsPath`].
    ///
    /// Normalisation collapses runs of `/` (`a//b` → `a/b`) and strips a
    /// trailing `/` (`a/b/` → `a/b`); everything else is preserved
    /// byte-for-byte. Rejected forms produce a [`VfsPathError`] without
    /// constructing the value.
    pub fn new(input: impl Into<String>) -> Result<Self, VfsPathError> {
        let input = input.into();
        if input.is_empty() {
            return Err(VfsPathError::Empty);
        }
        if input.starts_with('/') {
            return Err(VfsPathError::Absolute);
        }
        if input.contains('\\') {
            return Err(VfsPathError::Backslash);
        }
        if input.contains(':') {
            return Err(VfsPathError::Colon);
        }
        if input.contains('\0') {
            return Err(VfsPathError::NulByte);
        }

        // Strip trailing slashes, then collapse repeated internal slashes.
        let trimmed = input.trim_end_matches('/');
        if trimmed.is_empty() {
            return Err(VfsPathError::Empty);
        }

        let mut normalised = String::with_capacity(trimmed.len());
        let mut last_was_slash = false;
        for ch in trimmed.chars() {
            if ch == '/' {
                if last_was_slash {
                    continue;
                }
                last_was_slash = true;
            } else {
                last_was_slash = false;
            }
            normalised.push(ch);
        }

        // Segment-level checks run after slash collapsing so we don't have
        // to think about empty segments.
        for segment in normalised.split('/') {
            match segment {
                ".." => return Err(VfsPathError::ParentTraversal),
                "." => return Err(VfsPathError::CurrentDir),
                "" => unreachable!("repeated slashes already collapsed"),
                _ => {}
            }
        }

        Ok(VfsPath(normalised))
    }

    /// Borrow the normalised path string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// File extension (excluding the dot), if the final segment has one.
    /// Returns `None` for paths with no `.` in their final segment, or with
    /// an empty extension (`foo.`).
    pub fn extension(&self) -> Option<&str> {
        let last = self.0.rsplit('/').next()?;
        let (_, ext) = last.rsplit_once('.')?;
        if ext.is_empty() { None } else { Some(ext) }
    }

    /// Returns `true` if `self` is `prefix` itself, or a descendant of it
    /// (i.e. `self.as_str()` starts with `prefix.as_str() + "/"`). Used by
    /// the definitions loader to scope a listing to a sub-prefix.
    pub fn is_within(&self, prefix: &VfsPath) -> bool {
        if self == prefix {
            return true;
        }
        let p = prefix.as_str();
        let s = self.as_str();
        s.len() > p.len() + 1 && s.starts_with(p) && s.as_bytes()[p.len()] == b'/'
    }
}

impl fmt::Display for VfsPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for VfsPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> VfsPath {
        VfsPath::new(s).expect("valid path")
    }

    fn err(s: &str) -> VfsPathError {
        VfsPath::new(s).expect_err("invalid path")
    }

    #[test]
    fn accepts_simple_relative_paths() {
        assert_eq!(ok("foo").as_str(), "foo");
        assert_eq!(ok("foo/bar").as_str(), "foo/bar");
        assert_eq!(ok("kinds/oak_tree.ron").as_str(), "kinds/oak_tree.ron");
        assert_eq!(ok("a/b/c/d.png").as_str(), "a/b/c/d.png");
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(err(""), VfsPathError::Empty);
        // All-slashes input collapses to empty after trimming.
        assert_eq!(err("///"), VfsPathError::Absolute);
    }

    #[test]
    fn rejects_absolute() {
        assert_eq!(err("/foo"), VfsPathError::Absolute);
        assert_eq!(err("/"), VfsPathError::Absolute);
        assert_eq!(err("/foo/bar"), VfsPathError::Absolute);
    }

    #[test]
    fn rejects_backslash() {
        assert_eq!(err(r"foo\bar"), VfsPathError::Backslash);
        assert_eq!(err(r"C:\foo"), VfsPathError::Backslash);
        assert_eq!(err(r"\foo"), VfsPathError::Backslash);
    }

    #[test]
    fn rejects_drive_letter() {
        // The check is "any colon", so it catches drive letters by structure
        // rather than parsing them specifically.
        assert_eq!(err("C:/foo"), VfsPathError::Colon);
        assert_eq!(err("c:foo"), VfsPathError::Colon);
        assert_eq!(err("foo:bar"), VfsPathError::Colon);
    }

    #[test]
    fn rejects_nul_byte() {
        assert_eq!(err("foo\0bar"), VfsPathError::NulByte);
    }

    #[test]
    fn rejects_parent_traversal() {
        assert_eq!(err(".."), VfsPathError::ParentTraversal);
        assert_eq!(err("../foo"), VfsPathError::ParentTraversal);
        assert_eq!(err("foo/.."), VfsPathError::ParentTraversal);
        assert_eq!(err("foo/../bar"), VfsPathError::ParentTraversal);
        // `..foo` is a real filename with two leading dots, not parent
        // traversal — only an exact `..` segment is rejected.
        assert_eq!(ok("..foo").as_str(), "..foo");
        assert_eq!(ok("foo..bar").as_str(), "foo..bar");
    }

    #[test]
    fn rejects_current_dir_segment() {
        assert_eq!(err("."), VfsPathError::CurrentDir);
        assert_eq!(err("./foo"), VfsPathError::CurrentDir);
        assert_eq!(err("foo/."), VfsPathError::CurrentDir);
        assert_eq!(err("foo/./bar"), VfsPathError::CurrentDir);
        // A `.` *within* a segment (like a file extension) is fine.
        assert_eq!(ok("foo.ron").as_str(), "foo.ron");
        assert_eq!(ok("a.b.c").as_str(), "a.b.c");
    }

    #[test]
    fn collapses_repeated_slashes() {
        assert_eq!(ok("foo//bar").as_str(), "foo/bar");
        assert_eq!(ok("foo///bar////baz").as_str(), "foo/bar/baz");
    }

    #[test]
    fn strips_trailing_slash() {
        assert_eq!(ok("foo/").as_str(), "foo");
        assert_eq!(ok("foo/bar/").as_str(), "foo/bar");
        // Combined with collapsing: `foo//` → `foo`.
        assert_eq!(ok("foo//").as_str(), "foo");
    }

    #[test]
    fn normalisation_is_idempotent() {
        let once = ok("foo//bar/");
        let twice = VfsPath::new(once.as_str()).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn extension_returns_final_dotted_suffix() {
        assert_eq!(ok("foo.ron").extension(), Some("ron"));
        assert_eq!(ok("kinds/oak.ron").extension(), Some("ron"));
        assert_eq!(ok("a.b.c").extension(), Some("c"));
        assert_eq!(ok("foo").extension(), None);
        // Earlier segment's dot doesn't count.
        assert_eq!(ok("kinds.v2/oak").extension(), None);
    }

    #[test]
    fn is_within_matches_self_and_descendants() {
        let kinds = ok("kinds");
        assert!(ok("kinds").is_within(&kinds));
        assert!(ok("kinds/oak.ron").is_within(&kinds));
        assert!(ok("kinds/sub/oak.ron").is_within(&kinds));
        // Sibling prefix that shares characters isn't a descendant.
        assert!(!ok("kindsx/oak.ron").is_within(&kinds));
        assert!(!ok("other/oak.ron").is_within(&kinds));
    }
}
