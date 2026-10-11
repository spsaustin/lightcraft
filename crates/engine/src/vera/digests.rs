//! The identity lane's memory: one fingerprint per picture, kept in memory and cached beside the library.
//!
//! A fingerprint is the picture's PDQ + pHash digests and, for a file with a CPNP-1 path (JPEG, PNG, WebP, TIFF, …),
//! its VER `pixel_hash` and `content_hash`: computed over the **Canonical Buffer** at full size (`lightcraft-ver`,
//! VER 1.0 §4–5 and errata E22: a perceptual digest is computed over the Canonical Buffer, never over a re-decoded or
//! resized intermediate). A file without a CPNP-1 path (a camera raw, a demo scene, any picture in the browser build)
//! gets **provisional** digests from its thumbnail-level picture, sRGB-encoded so they compare with the canonical ones,
//! and no `pixel_hash`.
//!
//! Each picture is fingerprinted once and remembered, keyed by its content key ([`crate::media::content_key`]: the
//! file's content hash, so a virtual copy or a re-linked file shares its original's fingerprint). The cache file
//! (`vera-digests.bin` in the library folder) is rebuildable: a damaged one, an older format or a foreign file is
//! ignored and written again. Fingerprints are kept out of the catalog on purpose: they are derived, small, and the
//! rendition a provisional one is computed from may change. The file lives where the library's other files do
//! ([`Sidecars`]: the library folder, or the browser's storage); new records are appended, and the file is written
//! whole only when records were dropped (photos removed, a torn tail).
//!
//! Layout (little-endian): `"LCVD"`, format version `2`, the digest algorithm's version
//! ([`lightcraft_vera::ALGORITHM_VERSION`]: a file made by another computation is ignored and rebuilt), then records of
//! `key_len u16`, the key (UTF-8), `pdq [u8; 32]`, `pdq_quality u8`, `phash u64`, `source u8` (what the digests were
//! computed from: [`SOURCE_CANONICAL`] or [`SOURCE_PROVISIONAL`]; records of any other source are dropped), `flags u8`
//! (bit 0: a content hash follows), then `pixel_hash [u8; 32]` for a canonical record and `content_hash [u8; 32]` when
//! flagged. Version 1 files (thumbnail-level digests in linear light) are rebuilt.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use lightcraft_vera::Digest;

use super::sidecar::{Loaded, Sidecars};

/// The cache file's name in the library folder.
pub(crate) const FILE_NAME: &str = "vera-digests.bin";
const MAGIC: &[u8; 4] = b"LCVD";
const VERSION: u8 = 2;
/// Digests over the full-size CPNP-1 Canonical Buffer, with its `pixel_hash`.
pub(crate) const SOURCE_CANONICAL: u8 = 2;
/// Provisional digests from the photo's thumbnail-level source (at most 512 px), sRGB-encoded: a file without a
/// CPNP-1 path.
pub(crate) const SOURCE_PROVISIONAL: u8 = 3;
/// Largest cache file read.
const MAX_FILE: u64 = 1 << 30;
/// Longest key stored (a content hash is 32 characters; a `file:` key carries a path).
const MAX_KEY: usize = 4096;
/// Bytes of a record after its key, before the optional hashes.
const TAIL: usize = 32 + 1 + 8 + 1 + 1;

/// One picture's fingerprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fingerprint {
    pub digest: Digest,
    /// The VER `pixel_hash` (SHA-256 of the Canonical Buffer); `None` for a provisional fingerprint.
    pub pixel: Option<[u8; 32]>,
    /// The VER `content_hash` (SHA-256 of the file's bytes), when the file was read.
    pub content: Option<[u8; 32]>,
}

fn hash32(hex: &str) -> Option<[u8; 32]> {
    let v = crate::ver::util::unhex(hex)?;
    v.try_into().ok()
}

impl Fingerprint {
    /// Provisional digests (no CPNP-1 path).
    #[cfg(test)]
    pub fn provisional(digest: Digest) -> Fingerprint {
        Fingerprint { digest, pixel: None, content: None }
    }

    /// The fingerprint of a VER identity (`quality`: PDQ's quality figure of the Canonical Buffer, `None` when unknown:
    /// then the PDQ digest is trusted, as a Record's own digest is). `None` when a value is not the hex it should be.
    pub fn from_identity(id: &lightcraft_ver::hashing::Identity, quality: Option<u8>) -> Option<Fingerprint> {
        let pdq = hash32(&id.pdq)?;
        let phash = u64::from_str_radix(&id.phash, 16).ok().filter(|_| id.phash.len() == 16)?;
        Some(Fingerprint {
            digest: Digest { pdq, pdq_quality: quality.unwrap_or(100).min(100), phash },
            pixel: Some(hash32(&id.pixel_hash)?),
            content: Some(hash32(&id.content_hash)?),
        })
    }

    pub fn is_provisional(&self) -> bool {
        self.pixel.is_none()
    }
}

#[derive(Default)]
pub(crate) struct DigestStore {
    map: HashMap<String, Fingerprint>,
    /// Keys whose digests are not in the file yet.
    unsaved: Vec<String>,
    files: Sidecars,
    /// The file on disk has this format's header and whole records, so new records can be appended to it.
    file_matches: bool,
}

impl DigestStore {
    /// Where the cache file is kept.
    pub fn files(&self) -> &Sidecars {
        &self.files
    }

    /// The cache file's path, for a library on disk.
    pub fn path(&self) -> Option<PathBuf> {
        self.files.path_of(FILE_NAME)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&Digest> {
        self.map.get(key).map(|f| &f.digest)
    }

    pub fn fingerprint(&self, key: &str) -> Option<&Fingerprint> {
        self.map.get(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Remember `key`'s provisional digest (see [`DigestStore::insert_fingerprint`]).
    #[cfg(test)]
    pub fn insert(&mut self, key: String, digest: Digest) {
        self.insert_fingerprint(key, Fingerprint::provisional(digest));
    }

    /// Remember `key`'s fingerprint (written to the file on the next [`DigestStore::save`]). A key too long to store is
    /// kept in memory only.
    pub fn insert_fingerprint(&mut self, key: String, fp: Fingerprint) {
        if self.map.get(&key) == Some(&fp) {
            return;
        }
        if key.len() <= MAX_KEY {
            self.unsaved.push(key.clone());
        }
        self.map.insert(key, fp);
    }

    pub fn has_unsaved(&self) -> bool {
        !self.unsaved.is_empty()
    }

    /// Start over with the cache file kept in `files` (what was not saved yet is written to the old file first). With
    /// `live`, records whose key is not in it (photos removed since) are dropped, and the file is rewritten without them.
    pub fn reset(&mut self, files: Sidecars, live: Option<&HashSet<String>>) {
        if let Err(e) = self.save() {
            log::warn!("vera: could not save the picture digests: {e}");
        }
        self.map.clear();
        self.unsaved.clear();
        self.file_matches = false;
        self.files = files;
        self.load(live);
    }

    fn load(&mut self, live: Option<&HashSet<String>>) {
        let bytes = match self.files.read(FILE_NAME, MAX_FILE) {
            Loaded::Missing => return,
            Loaded::TooLarge => {
                log::warn!("vera: {} is too large to be a digest cache; it will be rebuilt", self.files.describe(FILE_NAME));
                return;
            }
            Loaded::Bytes(b) => b,
        };
        let (map, whole) = parse(&bytes);
        let Some(map) = map else { return };
        let read = map.len();
        let mut map = map;
        if let Some(live) = live {
            map.retain(|k, _| live.contains(k));
        }
        // anything dropped, or a torn last record, makes the file be written whole on the next save: appending after a
        // torn record would misalign every record after it
        self.file_matches = whole && map.len() == read;
        if !self.file_matches {
            log::debug!("vera: digest cache {}: {} of {read} records kept; rewriting", self.files.describe(FILE_NAME), map.len());
            self.unsaved = map.keys().cloned().collect();
        }
        self.map = map;
    }

    /// Write what was added since the last save: appended when the file already has this format, else the whole store is
    /// written fresh (atomically). A failure is returned and the records stay pending.
    pub fn save(&mut self) -> std::io::Result<()> {
        if !self.files.persists() {
            self.unsaved.clear();
            return Ok(());
        }
        if self.unsaved.is_empty() {
            return Ok(());
        }
        if self.file_matches {
            let mut buf = Vec::new();
            for k in &self.unsaved {
                if let Some(d) = self.map.get(k) {
                    record(&mut buf, k, d);
                }
            }
            match self.files.append(FILE_NAME, &buf) {
                Ok(()) => {
                    self.unsaved.clear();
                    return Ok(());
                }
                // the file went away (or cannot be appended to): write it whole
                Err(e) => log::debug!("vera: appending to {}: {e}; rewriting", self.files.describe(FILE_NAME)),
            }
        }
        let mut buf = Vec::with_capacity(6 + self.map.len() * (TAIL + 34 + 64));
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.push(lightcraft_vera::ALGORITHM_VERSION);
        for (k, d) in &self.map {
            if k.len() <= MAX_KEY {
                record(&mut buf, k, d);
            }
        }
        self.files.write_atomic(FILE_NAME, &buf).map_err(std::io::Error::other)?;
        self.file_matches = true;
        self.unsaved.clear();
        Ok(())
    }

    /// Size of the cache file, when there is one.
    pub fn file_len(&self) -> Option<u64> {
        self.files.len(FILE_NAME)
    }
}

impl Drop for DigestStore {
    fn drop(&mut self) {
        let _ = self.save();
    }
}

fn record(buf: &mut Vec<u8>, key: &str, f: &Fingerprint) {
    let Ok(len) = u16::try_from(key.len()) else { return };
    let d = &f.digest;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&d.pdq);
    buf.push(d.pdq_quality);
    buf.extend_from_slice(&d.phash.to_le_bytes());
    buf.push(if f.pixel.is_some() { SOURCE_CANONICAL } else { SOURCE_PROVISIONAL });
    buf.push(u8::from(f.content.is_some()));
    if let Some(p) = &f.pixel {
        buf.extend_from_slice(p);
    }
    if let Some(c) = &f.content {
        buf.extend_from_slice(c);
    }
}

/// The records of a cache file: `None` for a file that is not one (wrong magic or version); otherwise the fingerprints
/// and whether the file ended on a record boundary. Records of an unknown source, with a key that is not UTF-8, or a
/// quality over 100, are left out.
fn parse(bytes: &[u8]) -> (Option<HashMap<String, Fingerprint>>, bool) {
    let Some(mut rest) = bytes
        .strip_prefix(MAGIC)
        .and_then(|r| r.split_first())
        .filter(|(v, _)| **v == VERSION)
        .and_then(|(_, r)| r.split_first())
        .filter(|(a, _)| **a == lightcraft_vera::ALGORITHM_VERSION)
        .map(|(_, r)| r)
    else {
        return (None, false);
    };
    let mut map = HashMap::new();
    let mut clean = true;
    while !rest.is_empty() {
        let Some((len, tail)) = rest.split_first_chunk::<2>() else {
            return (Some(map), false);
        };
        let len = usize::from(u16::from_le_bytes(*len));
        let Some((key, tail)) = tail.split_at_checked(len) else {
            return (Some(map), false);
        };
        let Some((body, tail)) = tail.split_first_chunk::<TAIL>() else {
            return (Some(map), false);
        };
        let (pdq, more) = body.split_first_chunk::<32>().map(|(a, b)| (*a, b)).unwrap_or(([0; 32], &[]));
        let (quality, phash, source, flags) = match more {
            [q, p0, p1, p2, p3, p4, p5, p6, p7, s, f] => (*q, u64::from_le_bytes([*p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7]), *s, *f),
            _ => (0, 0, 0, 0),
        };
        let mut tail = tail;
        let mut take = |n: bool| -> Result<Option<[u8; 32]>, ()> {
            if !n {
                return Ok(None);
            }
            let (h, t) = tail.split_first_chunk::<32>().ok_or(())?;
            tail = t;
            Ok(Some(*h))
        };
        let (Ok(pixel), Ok(content)) = (take(source == SOURCE_CANONICAL), take(flags & 1 == 1)) else {
            return (Some(map), false);
        };
        rest = tail;
        match std::str::from_utf8(key) {
            Ok(k) if (source == SOURCE_CANONICAL || source == SOURCE_PROVISIONAL) && quality <= 100 && flags <= 1 && !k.is_empty() => {
                map.insert(k.to_string(), Fingerprint { digest: Digest { pdq, pdq_quality: quality, phash }, pixel, content });
            }
            // a record of another rendition, or damaged: dropped, and the file rewritten without it
            _ => clean = false,
        }
    }
    (Some(map), clean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lc-vera-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn d(n: u8) -> Digest {
        Digest { pdq: [n; 32], pdq_quality: 80, phash: u64::from(n) * 0x0101_0101 }
    }

    #[test]
    fn round_trip_append_and_reload() {
        let dir = temp_dir("roundtrip");
        let path = dir.join(FILE_NAME);
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        st.insert("aaa".into(), d(1));
        st.insert("bbb".into(), d(2));
        st.save().unwrap();
        let first = std::fs::metadata(&path).unwrap().len();
        // a second save appends only the new record
        st.insert("ccc".into(), d(3));
        assert!(st.has_unsaved());
        st.save().unwrap();
        let second = std::fs::metadata(&path).unwrap().len();
        assert_eq!(second - first, (2 + 3 + TAIL) as u64);
        // a canonical fingerprint carries its pixel and content hashes, and they come back
        let fp = Fingerprint { digest: d(9), pixel: Some([5; 32]), content: Some([6; 32]) };
        st.insert_fingerprint("ddd".into(), fp);
        st.save().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len() - second, (2 + 3 + TAIL + 64) as u64);
        drop(st);
        let mut again = DigestStore::default();
        again.reset(Sidecars::Dir(dir.clone()), None);
        assert_eq!(again.len(), 4);
        assert_eq!(again.get("bbb"), Some(&d(2)));
        assert_eq!(again.fingerprint("ddd"), Some(&fp));
        assert!(again.fingerprint("aaa").unwrap().is_provisional());
        assert_eq!(again.get("ccc"), Some(&d(3)));
        assert!(again.file_matches && !again.has_unsaved());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_and_truncated_files_never_panic() {
        let dir = temp_dir("corrupt");
        let path = dir.join(FILE_NAME);
        // random garbage, an empty file, a header alone, a huge key length
        for bytes in [b"garbage!".to_vec(), Vec::new(), b"LCVD\x02".to_vec(), b"LCVD\x02\x01\xff\xffab".to_vec(), b"LCVD\x03\x01\x00a".to_vec()] {
            std::fs::write(&path, &bytes).unwrap();
            let mut st = DigestStore::default();
            st.reset(Sidecars::Dir(dir.clone()), None);
            assert!(st.is_empty());
        }
        // a whole record then a torn one: the whole one is kept and the file is rewritten before anything is appended
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        st.insert("key1".into(), d(7));
        st.save().unwrap();
        drop(st);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[4, 0, b'k', b'e']);
        std::fs::write(&path, &bytes).unwrap();
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        assert_eq!(st.len(), 1);
        assert!(!st.file_matches);
        st.insert("key2".into(), d(8));
        st.save().unwrap();
        drop(st);
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        assert_eq!((st.len(), st.get("key2")), (2, Some(&d(8))));
        assert!(st.file_matches);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_file_is_rewritten() {
        let dir = temp_dir("foreign");
        let path = dir.join(FILE_NAME);
        std::fs::write(&path, b"LCFE\x02not ours at all").unwrap();
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        assert!(st.is_empty() && !st.file_matches);
        st.insert("k".into(), d(4));
        st.save().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(&[b'L', b'C', b'V', b'D', 2, lightcraft_vera::ALGORITHM_VERSION]), "{bytes:?}");
        drop(st);
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        assert_eq!(st.get("k"), Some(&d(4)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A version 1 cache (thumbnail-level digests in linear light) is not read: it is rebuilt over the Canonical Buffer.
    #[test]
    fn version_1_files_are_rebuilt() {
        let mut bytes = b"LCVD\x01".to_vec();
        bytes.push(lightcraft_vera::ALGORITHM_VERSION);
        bytes.extend_from_slice(&[1, 0, b'k']);
        bytes.extend_from_slice(&[0; 32 + 1 + 8 + 1]);
        let (map, clean) = parse(&bytes);
        assert!(map.is_none() && !clean);
    }

    #[test]
    fn records_of_another_rendition_are_dropped() {
        let mut bytes = b"LCVD\x02".to_vec();
        bytes.push(lightcraft_vera::ALGORITHM_VERSION);
        let mut ok = Vec::new();
        record(&mut ok, "new", &Fingerprint::provisional(d(1)));
        let mut old = ok.clone();
        // the same record, made from another rendition (the source byte sits before the flags)
        let at = old.len() - 2;
        old[at] = 9;
        bytes.extend_from_slice(&old);
        bytes.extend_from_slice(&ok);
        let (map, clean) = parse(&bytes);
        assert_eq!(map.unwrap().len(), 1);
        assert!(!clean);
    }

    #[test]
    fn compaction_drops_orphans() {
        let dir = temp_dir("compact");
        let path = dir.join(FILE_NAME);
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        for (k, n) in [("keep", 1), ("gone", 2), ("also", 3)] {
            st.insert(k.into(), d(n));
        }
        st.save().unwrap();
        drop(st);
        let full = std::fs::metadata(&path).unwrap().len();
        let live: HashSet<String> = ["keep".to_string(), "also".to_string()].into_iter().collect();
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), Some(&live));
        assert_eq!(st.len(), 2);
        assert!(!st.contains("gone"));
        // rewritten without the orphan on the next save
        st.save().unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() < full);
        drop(st);
        let mut st = DigestStore::default();
        st.reset(Sidecars::Dir(dir.clone()), None);
        assert_eq!(st.len(), 2);
        assert!(st.file_matches);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn in_memory_store_never_writes() {
        let mut st = DigestStore::default();
        st.insert("k".into(), d(1));
        st.save().unwrap();
        assert!(!st.has_unsaved());
        assert_eq!(st.len(), 1);
        assert_eq!(st.file_len(), None);
    }
}

#[cfg(test)]
mod algorithm_version_tests {
    use super::*;

    /// A cache made by another digest computation is ignored whole: comparing its records with fresh ones would be
    /// comparing two different fingerprints of the same pixels.
    #[test]
    fn another_algorithm_version_is_ignored() {
        let mut bytes = b"LCVD\x02".to_vec();
        bytes.push(lightcraft_vera::ALGORITHM_VERSION.wrapping_add(1));
        let (map, clean) = parse(&bytes);
        assert!(map.is_none());
        assert!(!clean);
    }
}
