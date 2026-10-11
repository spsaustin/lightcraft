//! The vector lane's memory: one image embedding per picture, kept in a [`lightcraft_vera::Space`] and cached beside
//! the library.
//!
//! An embedding costs a model pass, so each picture is embedded once and remembered, keyed by its content key
//! ([`crate::media::content_key`], like the digests: a virtual copy shares its original's vector). The space itself is
//! keyed by a `u64` derived from the content key ([`key_id`]); this store keeps the maps both ways. The cache file
//! (`vera-embeddings-<model id>.bin` in the library folder) is rebuildable: a damaged one, another format, another
//! model or another dimension is ignored and written again. It lives where the library's other files do ([`Sidecars`]:
//! the library folder, or the browser's storage); new records are appended (on the web too: only the new bytes are
//! flushed), and the file is written whole only when records were dropped.
//!
//! Layout (little-endian): `"LCVE"`, format version `1`, the model id (`u16` length + UTF-8), `dim u32`, then records
//! of `key_len u16`, the content key (UTF-8), `role u8` and `dim × f32`. Roles: [`ROLE_CANONICAL`] (`0`, the picture
//! as it is, rendered with default settings: what the space holds), [`ROLE_CONTEXTUAL`] (`1`, the text tower's
//! embedding of the photo's accepted description: "title. caption. keywords") and [`ROLE_FUSED`] (`2`, the normalised
//! sum of the two, which `photo.similar` prefers). The last two are kept beside the space, not in it.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

use lightcraft_vera::Space;
use lightcraft_vera::cutoff::Background;

use super::sidecar::{Opened, Sidecars};

const MAGIC: &[u8; 4] = b"LCVE";
const VERSION: u8 = 1;
/// The picture's own embedding (default settings, upright, uncropped).
pub(crate) const ROLE_CANONICAL: u8 = 0;
/// The text embedding of the photo's accepted description (Phase 3, written by `describe.accept`).
pub(crate) const ROLE_CONTEXTUAL: u8 = 1;
/// The normalised sum of the canonical and contextual vectors.
pub(crate) const ROLE_FUSED: u8 = 2;
/// Largest cache file read (a million pictures at 768 numbers).
const MAX_FILE: u64 = 4 << 30;
/// Longest key stored (a content hash is 32 characters; a `file:` key carries a path).
const MAX_KEY: usize = 4096;
/// Longest model id in a header.
const MAX_MODEL_ID: usize = 256;

/// The file name for `model_id`'s embeddings in the library folder.
pub(crate) fn file_name(model_id: &str) -> String {
    format!("vera-embeddings-{model_id}.bin")
}

/// The space key of a content key: the low 64 bits of its 128-bit content hash (stable across runs and platforms).
pub(crate) fn key_id(key: &str) -> u64 {
    lightcraft_preview::hash_bytes(key.as_bytes()).0 as u64
}

pub(crate) struct EmbeddingStore {
    model_id: String,
    dim: usize,
    /// The canonical vectors (role 0), by [`key_id`].
    space: Option<Space>,
    /// Space key → content key, and back.
    keys: HashMap<u64, String>,
    ids: HashMap<String, u64>,
    /// The contextual and fused vectors (unit length), by (content key, role).
    extra: HashMap<(String, u8), Vec<f32>>,
    /// Keys (and roles) whose vectors are not in the file yet.
    unsaved: Vec<(String, u8)>,
    files: Sidecars,
    /// The file on disk has this format's header and whole records, so new records can be appended to it.
    file_matches: bool,
    /// The space's background cosine statistics, sampled on first use and again once stale.
    background: Option<Background>,
}

impl EmbeddingStore {
    /// An empty, in-memory store for `model_id` vectors of `dim` numbers.
    pub fn new(model_id: &str, dim: usize) -> EmbeddingStore {
        EmbeddingStore {
            model_id: model_id.to_string(),
            dim,
            space: Space::new(model_id, dim),
            keys: HashMap::new(),
            ids: HashMap::new(),
            extra: HashMap::new(),
            unsaved: Vec::new(),
            files: Sidecars::Memory,
            file_matches: false,
            background: None,
        }
    }

    /// Where the cache file is kept.
    pub fn files(&self) -> &Sidecars {
        &self.files
    }

    /// The cache file's name.
    fn name(&self) -> String {
        file_name(&self.model_id)
    }

    pub fn len(&self) -> usize {
        self.space.as_ref().map_or(0, Space::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, key: &str) -> bool {
        self.ids.get(key).is_some_and(|id| self.space.as_ref().is_some_and(|s| s.contains(*id)))
    }

    /// The space key of a stored content key.
    pub fn id_of(&self, key: &str) -> Option<u64> {
        self.ids.get(key).copied()
    }

    /// The content key of a space key.
    pub fn key_of(&self, id: u64) -> Option<&str> {
        self.keys.get(&id).map(String::as_str)
    }

    /// `key`'s unit vector.
    pub fn get(&self, key: &str) -> Option<&[f32]> {
        let id = self.ids.get(key)?;
        self.space.as_ref()?.get(*id)
    }

    pub fn space(&self) -> Option<&Space> {
        self.space.as_ref()
    }

    /// Remember `key`'s vector (written to the file on the next [`EmbeddingStore::save`]). `false` when it cannot be
    /// kept: the wrong length, not finite, zero, or another key already holds its space key. A key too long to store
    /// is kept in memory only.
    pub fn insert(&mut self, key: String, vector: &[f32]) -> bool {
        let id = key_id(&key);
        if self.keys.get(&id).is_some_and(|k| *k != key) {
            log::warn!("vera: two pictures share an embedding key; the second is not embedded");
            return false;
        }
        let Some(space) = self.space.as_mut() else { return false };
        if !space.insert(id, vector) {
            return false;
        }
        if key.len() <= MAX_KEY {
            self.unsaved.push((key.clone(), ROLE_CANONICAL));
        }
        self.keys.insert(id, key.clone());
        self.ids.insert(key, id);
        true
    }

    /// Remember `key`'s vector of `role` ([`ROLE_CANONICAL`] goes to the space, as [`EmbeddingStore::insert`]; the
    /// others are normalised to unit length and kept beside it). `false` for a vector that cannot be kept (the wrong
    /// length, not finite, zero) or an unknown role.
    pub fn insert_role(&mut self, key: String, role: u8, vector: &[f32]) -> bool {
        if role == ROLE_CANONICAL {
            return self.insert(key, vector);
        }
        if !matches!(role, ROLE_CONTEXTUAL | ROLE_FUSED) || vector.len() != self.dim || key.is_empty() {
            return false;
        }
        let Some(v) = unit(vector) else { return false };
        if key.len() <= MAX_KEY {
            self.unsaved.push((key.clone(), role));
        }
        self.extra.insert((key, role), v);
        true
    }

    /// `key`'s vector of `role` (unit length).
    pub fn get_role(&self, key: &str, role: u8) -> Option<&[f32]> {
        if role == ROLE_CANONICAL {
            return self.get(key);
        }
        self.extra.get(&(key.to_string(), role)).map(Vec::as_slice)
    }

    /// How many photos have a vector of `role` beside the space.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn role_count(&self, role: u8) -> usize {
        self.extra.keys().filter(|(_, r)| *r == role).count()
    }

    pub fn has_unsaved(&self) -> bool {
        !self.unsaved.is_empty()
    }

    /// The background of the space for the noise cutoff: sampled on first use, and again once the space has moved
    /// enough since ([`Space::background_is_stale`]). `None` under two vectors.
    pub fn background(&mut self) -> Option<Background> {
        let space = self.space.as_ref()?;
        let stale = self.background.as_ref().is_none_or(|bg| space.background_is_stale(bg));
        if stale {
            self.background = space.background();
        }
        self.background
    }

    /// Start over with the cache file kept in `files` (what was not saved yet is written to the old file first). With
    /// `live`, records whose key is not in it (photos removed since) are dropped, and the file is rewritten without them.
    pub fn reset(&mut self, files: Sidecars, live: Option<&HashSet<String>>) {
        if let Err(e) = self.save() {
            log::warn!("vera: could not save the picture embeddings: {e}");
        }
        self.clear();
        self.files = files;
        self.load(live);
    }

    fn clear(&mut self) {
        self.space = Space::new(&self.model_id, self.dim);
        self.keys.clear();
        self.ids.clear();
        self.extra.clear();
        self.unsaved.clear();
        self.file_matches = false;
        self.background = None;
    }

    /// Forget every vector and delete the cache file (`vera.model.remove`). `Ok(true)` when there was a file.
    pub fn delete_file(&mut self) -> std::io::Result<bool> {
        self.clear();
        self.files.remove(&self.name())
    }

    fn load(&mut self, live: Option<&HashSet<String>>) {
        let name = self.name();
        let mut reader = match self.files.open(&name, MAX_FILE) {
            Opened::Missing => return,
            Opened::TooLarge => {
                log::warn!("vera: {} is too large to be an embedding cache; it will be rebuilt", self.files.describe(&name));
                return;
            }
            Opened::Reader(r) => r,
        };
        let Some(parsed) = parse(&mut reader, &self.model_id, self.dim) else {
            log::debug!("vera: {} is not an embedding cache of {} at {}; it will be rebuilt", self.files.describe(&name), self.model_id, self.dim);
            return;
        };
        let read = parsed.records.len();
        let mut kept = 0usize;
        let mut clean = parsed.whole;
        for (key, role, vector) in parsed.records {
            if live.is_some_and(|l| !l.contains(&key)) {
                continue;
            }
            // a later record of the same key and role replaces an earlier one
            let was = if role == ROLE_CANONICAL { self.contains(&key) } else { self.extra.contains_key(&(key.clone(), role)) };
            if self.insert_role(key, role, &vector) {
                if !was {
                    kept += 1;
                }
            } else {
                clean = false;
            }
        }
        // anything dropped, or a torn last record, makes the file be written whole on the next save: appending after a
        // torn record would misalign every record after it
        self.file_matches = clean && kept == read;
        if self.file_matches {
            self.unsaved.clear();
        } else {
            log::debug!("vera: embedding cache {}: {kept} of {read} records kept; rewriting", self.files.describe(&name));
            self.unsaved = self.ids.keys().map(|k| (k.clone(), ROLE_CANONICAL)).chain(self.extra.keys().cloned()).collect();
        }
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
        let name = self.name();
        if self.file_matches {
            let mut buf = Vec::new();
            for (k, role) in &self.unsaved {
                if let Some(v) = self.get_role(k, *role) {
                    record(&mut buf, k, *role, v);
                }
            }
            match self.files.append(&name, &buf) {
                Ok(()) => {
                    self.unsaved.clear();
                    return Ok(());
                }
                // the file went away (or cannot be appended to): write it whole
                Err(e) => log::debug!("vera: appending to {}: {e}; rewriting", self.files.describe(&name)),
            }
        }
        self.write_whole(&name)?;
        self.file_matches = true;
        self.unsaved.clear();
        Ok(())
    }

    /// The whole store, written atomically (on disk: next to the file and renamed over it, so a failed write never
    /// leaves half a file).
    fn write_whole(&self, name: &str) -> std::io::Result<()> {
        self.files.write_streamed(name, "bin.part", &mut |w: &mut dyn Write| {
            w.write_all(&header(&self.model_id, self.dim))?;
            let mut buf = Vec::new();
            if let Some(space) = &self.space {
                for (id, v) in space.iter() {
                    let Some(k) = self.keys.get(&id).filter(|k| k.len() <= MAX_KEY) else { continue };
                    buf.clear();
                    record(&mut buf, k, ROLE_CANONICAL, v);
                    w.write_all(&buf)?;
                }
            }
            let mut extra: Vec<(&(String, u8), &Vec<f32>)> = self.extra.iter().filter(|((k, _), _)| k.len() <= MAX_KEY).collect();
            extra.sort_by(|a, b| a.0.cmp(b.0));
            for ((k, role), v) in extra {
                buf.clear();
                record(&mut buf, k, *role, v);
                w.write_all(&buf)?;
            }
            Ok(())
        })
    }

    /// Size of the cache file, when there is one.
    pub fn file_len(&self) -> Option<u64> {
        self.files.len(&self.name())
    }
}

/// `v` scaled to unit length; `None` when it is not finite or zero.
pub(crate) fn unit(v: &[f32]) -> Option<Vec<f32>> {
    let n = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if !n.is_finite() || n <= 0.0 {
        return None;
    }
    Some(v.iter().map(|x| (f64::from(*x) / n) as f32).collect())
}

impl Drop for EmbeddingStore {
    fn drop(&mut self) {
        let _ = self.save();
    }
}

fn header(model_id: &str, dim: usize) -> Vec<u8> {
    let mut h = Vec::with_capacity(4 + 1 + 2 + model_id.len() + 4);
    h.extend_from_slice(MAGIC);
    h.push(VERSION);
    let id = model_id.as_bytes().get(..model_id.len().min(MAX_MODEL_ID)).unwrap_or_default();
    h.extend_from_slice(&(id.len() as u16).to_le_bytes());
    h.extend_from_slice(id);
    h.extend_from_slice(&u32::try_from(dim).unwrap_or(0).to_le_bytes());
    h
}

fn record(buf: &mut Vec<u8>, key: &str, role: u8, v: &[f32]) {
    let Ok(len) = u16::try_from(key.len()) else { return };
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.push(role);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

struct Parsed {
    /// Records in file order, with their roles (a key and role may repeat; the last one wins).
    records: Vec<(String, u8, Vec<f32>)>,
    /// The file ended on a record boundary and every record was readable.
    whole: bool,
}

/// Read as much of `r` as fills `buf`; how many bytes that was (fewer only at the end of the file).
fn fill(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(buf.get_mut(n..).unwrap_or_default()) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

/// The records of a cache file of `model_id` at `dim`: `None` for a file that is not one (wrong magic, version, model
/// or dimension). A record with a key that is not UTF-8 or an unknown role is dropped (the file is then not whole).
fn parse(r: &mut impl Read, model_id: &str, dim: usize) -> Option<Parsed> {
    let mut head = [0u8; 7];
    if fill(r, &mut head).ok()? != head.len() {
        return None;
    }
    let (magic, rest) = head.split_first_chunk::<4>()?;
    if magic != MAGIC || rest.first() != Some(&VERSION) {
        return None;
    }
    let id_len = usize::from(u16::from_le_bytes([*rest.get(1)?, *rest.get(2)?]));
    if id_len > MAX_MODEL_ID {
        return None;
    }
    let mut id = vec![0u8; id_len];
    if fill(r, &mut id).ok()? != id_len || id != model_id.as_bytes() {
        return None;
    }
    let mut d = [0u8; 4];
    if fill(r, &mut d).ok()? != 4 || usize::try_from(u32::from_le_bytes(d)).ok()? != dim || dim == 0 {
        return None;
    }
    let body_len = dim.checked_mul(4)?;
    let mut records = Vec::new();
    let mut whole = true;
    let mut body = vec![0u8; body_len];
    loop {
        let mut len = [0u8; 2];
        match fill(r, &mut len) {
            Ok(0) => break,
            Ok(2) => {}
            _ => {
                whole = false;
                break;
            }
        }
        let key_len = usize::from(u16::from_le_bytes(len));
        let mut key = vec![0u8; key_len];
        let mut role = [0u8; 1];
        if fill(r, &mut key).ok() != Some(key_len) || fill(r, &mut role).ok() != Some(1) || fill(r, &mut body).ok() != Some(body_len) {
            whole = false;
            break;
        }
        let [role] = role;
        match (std::str::from_utf8(&key), role) {
            (Ok(k), ROLE_CANONICAL | ROLE_CONTEXTUAL | ROLE_FUSED) if !k.is_empty() => {
                let v: Vec<f32> = body.chunks_exact(4).filter_map(|c| c.try_into().ok().map(f32::from_le_bytes)).collect();
                records.push((k.to_string(), role, v));
            }
            _ => whole = false,
        }
    }
    Some(Parsed { records, whole })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const DIM: usize = 8;
    const MODEL: &str = "test-model";

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lc-vera-emb-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn v(n: usize) -> Vec<f32> {
        (0..DIM).map(|i| if i == n % DIM { 1.0 } else { 0.1 * (i as f32) }).collect()
    }

    fn store(path: &Path) -> EmbeddingStore {
        let mut st = EmbeddingStore::new(MODEL, DIM);
        st.reset(Sidecars::Dir(path.parent().unwrap().to_path_buf()), None);
        st
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        let n = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter().zip(b).all(|(x, y)| (x - y / n).abs() < 1e-6)
    }

    #[test]
    fn round_trip_append_and_reload() {
        let dir = temp_dir("roundtrip");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        assert!(st.insert("aaa".into(), &v(1)));
        assert!(st.insert("bbb".into(), &v(2)));
        st.save().unwrap();
        let first = std::fs::metadata(&path).unwrap().len();
        assert_eq!(first as usize, header(MODEL, DIM).len() + 2 * (2 + 3 + 1 + DIM * 4));
        // a second save appends only the new record
        assert!(st.insert("ccc".into(), &v(3)));
        assert!(st.has_unsaved());
        st.save().unwrap();
        let second = std::fs::metadata(&path).unwrap().len();
        assert_eq!(second - first, (2 + 3 + 1 + DIM * 4) as u64);
        drop(st);
        let again = store(&path);
        assert_eq!(again.len(), 3);
        assert!(close(again.get("bbb").unwrap(), &v(2)));
        assert!(close(again.get("ccc").unwrap(), &v(3)));
        assert!(again.file_matches && !again.has_unsaved());
        assert_eq!(again.key_of(key_id("aaa")), Some("aaa"));
        assert_eq!(again.id_of("aaa"), Some(key_id("aaa")));
        // the wrong length, zeros and non-numbers are refused
        let mut again = again;
        assert!(!again.insert("bad".into(), &[1.0; 3]));
        assert!(!again.insert("zero".into(), &[0.0; DIM]));
        assert!(!again.insert("nan".into(), &[f32::NAN; DIM]));
        assert_eq!(again.len(), 3);
        drop(again);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_headers_are_ignored_and_rewritten() {
        let dir = temp_dir("foreign");
        let path = dir.join(file_name(MODEL));
        let mut other_model = header("another-model", DIM);
        record(&mut other_model, "k", ROLE_CANONICAL, &v(1));
        let mut other_dim = header(MODEL, DIM * 2);
        record(&mut other_dim, "k", ROLE_CANONICAL, &[1.0; DIM * 2]);
        let mut other_version = header(MODEL, DIM);
        other_version[4] = 2;
        for bytes in [b"garbage!".to_vec(), Vec::new(), b"LCVE".to_vec(), b"LCVE\x01\xff\xff".to_vec(), other_model, other_dim, other_version] {
            std::fs::write(&path, &bytes).unwrap();
            let mut st = store(&path);
            assert!(st.is_empty() && !st.file_matches, "{bytes:?}");
            assert!(st.insert("mine".into(), &v(4)));
            st.save().unwrap();
            drop(st);
            let written = std::fs::read(&path).unwrap();
            assert!(written.starts_with(&header(MODEL, DIM)), "rewritten with this model's header");
            let st = store(&path);
            assert_eq!(st.len(), 1);
            assert!(st.contains("mine") && !st.contains("k"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_tail_and_damaged_records_are_dropped() {
        let dir = temp_dir("torn");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        st.insert("key1".into(), &v(1));
        st.save().unwrap();
        drop(st);
        let mut bytes = std::fs::read(&path).unwrap();
        // a contextual record is read beside the space, an unknown role is damage, and the last record is cut short
        record(&mut bytes, "ctx", ROLE_CONTEXTUAL, &v(2));
        record(&mut bytes, "odd", 9, &v(3));
        let mut torn = Vec::new();
        record(&mut torn, "key9", ROLE_CANONICAL, &v(9));
        bytes.extend_from_slice(&torn[..torn.len() - 5]);
        std::fs::write(&path, &bytes).unwrap();
        let mut st = store(&path);
        assert_eq!(st.len(), 1);
        assert!(!st.file_matches, "rewritten before anything is appended");
        assert!(!st.contains("ctx") && !st.contains("odd") && !st.contains("key9"));
        assert!(st.get_role("ctx", ROLE_CONTEXTUAL).is_some(), "the contextual record is kept beside the space");
        st.insert("key2".into(), &v(2));
        st.save().unwrap();
        drop(st);
        let st = store(&path);
        assert_eq!(st.len(), 2);
        assert!(st.file_matches);
        // a huge key length never allocates past the file
        let mut huge = header(MODEL, DIM);
        huge.extend_from_slice(&[0xff, 0xff, b'a']);
        std::fs::write(&path, &huge).unwrap();
        let st = store(&path);
        assert!(st.is_empty());
        drop(st);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_drops_orphans() {
        let dir = temp_dir("compact");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        for (k, n) in [("keep", 1), ("gone", 2), ("also", 3)] {
            st.insert(k.into(), &v(n));
        }
        st.save().unwrap();
        drop(st);
        let full = std::fs::metadata(&path).unwrap().len();
        let live: HashSet<String> = ["keep".to_string(), "also".to_string()].into_iter().collect();
        let mut st = EmbeddingStore::new(MODEL, DIM);
        st.reset(Sidecars::Dir(dir.clone()), Some(&live));
        assert_eq!(st.len(), 2);
        assert!(!st.contains("gone"));
        st.save().unwrap();
        assert!(std::fs::metadata(&path).unwrap().len() < full);
        drop(st);
        let st = store(&path);
        assert_eq!(st.len(), 2);
        assert!(st.file_matches);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn contextual_and_fused_roles_round_trip() {
        let dir = temp_dir("roles");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        assert!(st.insert("pic".into(), &v(1)));
        assert!(st.insert_role("pic".into(), ROLE_CONTEXTUAL, &v(2)));
        assert!(st.insert_role("pic".into(), ROLE_FUSED, &v(3)));
        assert!(!st.insert_role("pic".into(), 7, &v(3)), "unknown role");
        assert!(!st.insert_role("pic".into(), ROLE_FUSED, &[0.0; DIM]), "zero");
        assert!(!st.insert_role("pic".into(), ROLE_FUSED, &[1.0; 3]), "wrong length");
        assert_eq!(st.len(), 1, "only the canonical vector is in the space");
        st.save().unwrap();
        // an appended role record
        assert!(st.insert_role("pic".into(), ROLE_CONTEXTUAL, &v(4)));
        st.save().unwrap();
        drop(st);
        let st = store(&path);
        assert_eq!((st.len(), st.role_count(ROLE_CONTEXTUAL), st.role_count(ROLE_FUSED)), (1, 1, 1));
        assert!(close(st.get_role("pic", ROLE_CONTEXTUAL).unwrap(), &v(4)), "the last one wins");
        assert!(close(st.get_role("pic", ROLE_FUSED).unwrap(), &v(3)));
        assert!(close(st.get_role("pic", ROLE_CANONICAL).unwrap(), &v(1)));
        drop(st);
        // compaction drops them with their picture
        let mut st = EmbeddingStore::new(MODEL, DIM);
        st.reset(Sidecars::Dir(dir.clone()), Some(&HashSet::new()));
        assert_eq!((st.len(), st.role_count(ROLE_CONTEXTUAL)), (0, 0));
        drop(st);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_repeated_key_keeps_its_last_vector() {
        let dir = temp_dir("repeat");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        st.insert("k".into(), &v(1));
        st.save().unwrap();
        st.insert("k".into(), &v(5));
        st.save().unwrap();
        drop(st);
        let st = store(&path);
        assert_eq!(st.len(), 1);
        assert!(close(st.get("k").unwrap(), &v(5)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn background_is_kept_until_the_space_moves() {
        let mut st = EmbeddingStore::new(MODEL, DIM);
        assert!(st.background().is_none(), "no background under two vectors");
        for i in 0..20 {
            st.insert(format!("k{i}"), &v(i));
        }
        let a = st.background().unwrap();
        assert_eq!(a.n_at_compute, 20);
        st.insert("one more".into(), &v(7));
        assert_eq!(st.background().unwrap().n_at_compute, 20, "one insert does not resample");
        for i in 0..5 {
            st.insert(format!("more{i}"), &v(i + 3));
        }
        assert_eq!(st.background().unwrap().n_at_compute, 26, "resampled once the space grew by over a tenth");
    }

    #[test]
    fn delete_and_in_memory() {
        let dir = temp_dir("delete");
        let path = dir.join(file_name(MODEL));
        let mut st = store(&path);
        st.insert("k".into(), &v(1));
        st.save().unwrap();
        assert!(path.is_file());
        assert!(st.delete_file().unwrap());
        assert!(!path.exists() && st.is_empty());
        assert!(!st.delete_file().unwrap());
        drop(st);
        assert!(!path.exists(), "nothing written back on drop");
        let mut mem = EmbeddingStore::new(MODEL, DIM);
        mem.insert("k".into(), &v(1));
        mem.save().unwrap();
        assert!(!mem.has_unsaved() && mem.file_len().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
