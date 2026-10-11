//! The vector lane's memory: one image embedding per picture, kept in a [`lightcraft_vera::Space`] and cached beside
//! the library.
//!
//! An embedding costs a model pass, so each picture is embedded once and remembered, keyed by its content key
//! ([`crate::media::content_key`], like the digests: a virtual copy shares its original's vector). The space itself is
//! keyed by a `u64` derived from the content key ([`key_id`]); this store keeps the maps both ways. The cache file
//! (`vera-embeddings-<model id>.bin` in the library folder) is rebuildable: a damaged one, another format, another
//! model or another dimension is ignored and written again.
//!
//! Layout (little-endian): `"LCVE"`, format version `1`, the model id (`u16` length + UTF-8), `dim u32`, then records
//! of `key_len u16`, the content key (UTF-8), `role u8` ([`ROLE_CANONICAL`]: the picture as it is, rendered with
//! default settings; `1` contextual and `2` fused are reserved for Phase 3 and skipped by this build) and `dim × f32`.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use lightcraft_vera::Space;
use lightcraft_vera::cutoff::Background;

const MAGIC: &[u8; 4] = b"LCVE";
const VERSION: u8 = 1;
/// The picture's own embedding (default settings, upright, uncropped).
pub(crate) const ROLE_CANONICAL: u8 = 0;
/// Reserved for Phase 3 (an embedding of the picture with its context, and a fusion of both): read past, not loaded.
const ROLE_RESERVED_MAX: u8 = 2;
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
    /// Keys whose vectors are not in the file yet.
    unsaved: Vec<String>,
    path: Option<PathBuf>,
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
            unsaved: Vec::new(),
            path: None,
            file_matches: false,
            background: None,
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
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
            self.unsaved.push(key.clone());
        }
        self.keys.insert(id, key.clone());
        self.ids.insert(key, id);
        true
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

    /// Start over with the cache file at `path` (what was not saved yet is written to the old file first). With `live`,
    /// records whose key is not in it (photos removed since) are dropped, and the file is rewritten without them.
    pub fn reset(&mut self, path: Option<PathBuf>, live: Option<&HashSet<String>>) {
        if let Err(e) = self.save() {
            log::warn!("vera: could not save the picture embeddings: {e}");
        }
        self.clear();
        self.path = path;
        if let Some(p) = self.path.clone() {
            self.load(&p, live);
        }
    }

    fn clear(&mut self) {
        self.space = Space::new(&self.model_id, self.dim);
        self.keys.clear();
        self.ids.clear();
        self.unsaved.clear();
        self.file_matches = false;
        self.background = None;
    }

    /// Forget every vector and delete the cache file (`vera.model.remove`). `Ok(true)` when there was a file.
    pub fn delete_file(&mut self) -> std::io::Result<bool> {
        self.clear();
        let Some(path) = &self.path else { return Ok(false) };
        match std::fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn load(&mut self, path: &Path, live: Option<&HashSet<String>>) {
        let Ok(meta) = std::fs::metadata(path) else { return };
        if meta.len() > MAX_FILE {
            log::warn!("vera: {} is too large to be an embedding cache; it will be rebuilt", path.display());
            return;
        }
        let Ok(file) = std::fs::File::open(path) else { return };
        let mut reader = BufReader::new(file);
        let Some(parsed) = parse(&mut reader, &self.model_id, self.dim) else {
            log::debug!("vera: {} is not an embedding cache of {} at {}; it will be rebuilt", path.display(), self.model_id, self.dim);
            return;
        };
        let read = parsed.records.len();
        let mut kept = 0usize;
        let mut clean = parsed.whole;
        for (key, vector) in parsed.records {
            if live.is_some_and(|l| !l.contains(&key)) {
                continue;
            }
            // a later record of the same key replaces an earlier one
            let was = self.contains(&key);
            if self.insert(key, &vector) {
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
            log::debug!("vera: embedding cache {}: {kept} of {read} records kept; rewriting", path.display());
            self.unsaved = self.ids.keys().cloned().collect();
        }
    }

    /// Write what was added since the last save: appended when the file already has this format, else the whole store is
    /// written fresh (atomically). A failure is returned and the records stay pending.
    pub fn save(&mut self) -> std::io::Result<()> {
        let Some(path) = self.path.clone() else {
            self.unsaved.clear();
            return Ok(());
        };
        if self.unsaved.is_empty() {
            return Ok(());
        }
        if self.file_matches {
            let mut buf = Vec::new();
            for k in &self.unsaved {
                if let Some(v) = self.get(k) {
                    record(&mut buf, k, ROLE_CANONICAL, v);
                }
            }
            match std::fs::OpenOptions::new().append(true).open(&path).and_then(|mut f| f.write_all(&buf)) {
                Ok(()) => {
                    self.unsaved.clear();
                    return Ok(());
                }
                // the file went away (or cannot be appended to): write it whole
                Err(e) => log::debug!("vera: appending to {}: {e}; rewriting", path.display()),
            }
        }
        self.write_whole(&path)?;
        self.file_matches = true;
        self.unsaved.clear();
        Ok(())
    }

    /// The whole store, written next to `path` and renamed over it (so a failed write never leaves half a file).
    fn write_whole(&self, path: &Path) -> std::io::Result<()> {
        let part = path.with_extension("bin.part");
        {
            let mut w = BufWriter::new(std::fs::File::create(&part)?);
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
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&part, path).inspect_err(|_| {
            let _ = std::fs::remove_file(&part);
        })
    }

    /// Size of the cache file on disk, when there is one.
    pub fn file_len(&self) -> Option<u64> {
        self.path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len())
    }
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
    /// Canonical records in file order (a key may repeat; the last one wins).
    records: Vec<(String, Vec<f32>)>,
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
/// or dimension). Records of the reserved roles are read past; a record with a key that is not UTF-8 or an unknown role
/// is dropped (the file is then not whole).
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
            (Ok(k), ROLE_CANONICAL) if !k.is_empty() => {
                let v: Vec<f32> = body.chunks_exact(4).filter_map(|c| c.try_into().ok().map(f32::from_le_bytes)).collect();
                records.push((k.to_string(), v));
            }
            // Phase 3's roles: not read by this build, kept in the file until it is rewritten
            (Ok(k), r) if r <= ROLE_RESERVED_MAX && !k.is_empty() => {}
            _ => whole = false,
        }
    }
    Some(Parsed { records, whole })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        st.reset(Some(path.to_path_buf()), None);
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
        // a reserved role is read past, an unknown one is damage, and the last record is cut short
        record(&mut bytes, "ctx", 1, &v(2));
        record(&mut bytes, "odd", 9, &v(3));
        let mut torn = Vec::new();
        record(&mut torn, "key9", ROLE_CANONICAL, &v(9));
        bytes.extend_from_slice(&torn[..torn.len() - 5]);
        std::fs::write(&path, &bytes).unwrap();
        let mut st = store(&path);
        assert_eq!(st.len(), 1);
        assert!(!st.file_matches, "rewritten before anything is appended");
        assert!(!st.contains("ctx") && !st.contains("odd") && !st.contains("key9"));
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
        st.reset(Some(path.clone()), Some(&live));
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
