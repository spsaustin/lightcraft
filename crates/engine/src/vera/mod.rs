//! VERA in the engine: *find by words* (the keyword lane) and *same picture* (the identity lane) over the library.
//!
//! The algorithms live in `lightcraft-vera` (pure functions, no I/O); this module decides what to digest and when, where
//! the results are kept and how the commands see them (`docs/vera.md`):
//! - [`digests`]: one PDQ + pHash digest per picture, keyed by content, cached in `vera-digests.bin` beside the library.
//! - The keyword index, built from the catalog when a search needs it and rebuilt when the catalog has changed since.
//! - [`IgnoredPairs`]: pictures the user said are *not* the same (`vera-ignored.json` beside the library). A user
//!   decision kept outside the op log for now: it is not undoable and does not travel with a catalog export.
//! - The background pump (`vera.pump`): a few digests per frame on the session thread, with a row in the activity stack.
//! - The import check: files about to be imported are compared with the library's digests ([`NearCheck`]).
//! - The vector lane ([`lane`], Phase 2): SigLIP 2 embeddings of every picture ([`embeddings`], made off the session
//!   thread by [`worker`]), fused with the keyword lane in `search.text`, read by `photo.similar` and by the `similar`
//!   tier of `photo.duplicates`; the model is downloaded on request (`vera.model.*`, [`model`]).
//! - The Describe lane ([`describe`], Phase 3): titles, captions, alt text and keywords proposed by an endpoint the
//!   user configures ([`remote`]) and by zero-shot keywords from the same model ([`keywords`]), waiting in a review
//!   queue ([`review`]) until the user accepts them.

pub(crate) mod commands;
pub(crate) mod describe;
pub(crate) mod digests;
pub(crate) mod embeddings;
pub(crate) mod keywords;
pub(crate) mod lane;
pub(crate) mod model;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod remote;
pub(crate) mod review;
pub(crate) mod vocabulary;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod worker;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lightcraft_catalog::{MediaKind, Photo, PhotoId};
use lightcraft_raster::Rgb32f;
use lightcraft_vera::{Digest, Index, Luma, Sameness};
use serde_json::{Value, json};
use web_time::Instant;

use crate::Session;
use crate::activity::{Cancel, TaskGuard};
use crate::import::{ImportCandidate, NearDuplicate};
use crate::media::{SourceLevel, content_key};

/// The user's "not the same picture" decisions, in the library folder.
pub(crate) const IGNORED_FILE: &str = "vera-ignored.json";
/// Largest ignored-pairs file read.
const MAX_IGNORED_FILE: usize = 1 << 20;
/// Most ignored pairs kept (well within the file cap).
const MAX_IGNORED: usize = 10_000;
/// Longest content key an ignored pair may name.
const MAX_PAIR_KEY: usize = 4096;
/// How often the pump writes new digests to the cache file (and whenever the work runs out).
const SAVE_EVERY: Duration = Duration::from_secs(3);
/// How often the pump may look through the catalog for photos without a digest (it is called every frame).
const REQUEUE_EVERY: Duration = Duration::from_secs(1);
/// Most import candidates decoded and compared in one review.
pub(crate) const MAX_IMPORT_CHECKS: usize = 2000;
/// Candidate digests remembered between reviews (by path and size).
const MAX_CANDIDATE_CACHE: usize = 20_000;

/// The keyword lane's fields, in the order [`fields`] gives them, and their weights: title, keywords, caption, alt
/// text, extended description, file stem, camera + lens + place.
pub(crate) const FIELD_WEIGHTS: [f32; 7] = [3.0, 2.5, 1.5, 1.0, 0.5, 1.0, 0.5];

/// VERA's preferences, saved with the library (`prefs.json` → `vera`).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct Prefs {
    /// Whether the vector lane runs (`None`: the default, on once the model is installed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_lane: Option<bool>,
    /// Where the Describe lane sends photos (`vera.describe.config`; the token is not here).
    #[serde(skip_serializing_if = "describe::Prefs::is_default")]
    pub describe: describe::Prefs,
}

/// How hard the background pump may work now (the app says, from what the user is doing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pace {
    Pause,
    Light,
    Normal,
    Full,
}

impl Pace {
    pub fn parse(name: &str) -> Option<Pace> {
        match name {
            "pause" => Some(Pace::Pause),
            "light" => Some(Pace::Light),
            "normal" => Some(Pace::Normal),
            "full" => Some(Pace::Full),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Pace::Pause => "pause",
            Pace::Light => "light",
            Pace::Normal => "normal",
            Pace::Full => "full",
        }
    }

    /// Pictures digested per call at this pace.
    fn per_call(self) -> usize {
        match self {
            Pace::Pause => 0,
            Pace::Light => 1,
            Pace::Normal => 2,
            Pace::Full => 4,
        }
    }
}

/// Pictures still to digest: one content key, and the photos that show it (the first that loads is used).
struct Pending {
    key: String,
    ids: Vec<PhotoId>,
}

/// VERA's session state ([`Session::vera`]).
#[derive(Default)]
pub(crate) struct State {
    pub digests: digests::DigestStore,
    /// The keyword index, as of catalog revision `index_rev`.
    pub index: Option<Index>,
    pub index_rev: u64,
    /// The pump's row in the activity stack, while there is work.
    pub task: Option<TaskGuard>,
    pub saved_at: Option<Instant>,
    pub ignored: IgnoredPairs,
    /// Photos without a digest yet, as of the last look (what a paused pump reports).
    pub pending_known: usize,
    /// The library folder the sidecar files were opened for (`Some(None)`: an in-memory library).
    opened: Option<Option<PathBuf>>,
    /// Pictures whose source could not be read this session (not retried every frame).
    pub skipped: HashSet<String>,
    /// The pump's work, taken from the end.
    queue: Vec<Pending>,
    queue_stamp: Option<u64>,
    queued_at: Option<Instant>,
    /// Most photos pending at once since the work last ran out (the progress bar's whole).
    peak: usize,
    /// Photos that can be digested, and how many of them are, as of the last look (kept up to date as the pump works).
    photos_known: usize,
    digested_known: usize,
    /// Import candidates' digests, by (path, file size).
    pub candidates: HashMap<(String, u64), Digest>,
    /// The vector lane: the model, the library's embeddings, their pump.
    pub lane: lane::Lane,
    /// The Describe lane: its queue, its worker threads, the review queue.
    pub describe: describe::Describe,
}

impl State {
    /// Another library was opened: what was learned is saved (dropping the stores writes them) and nothing carries over
    /// but the vector lane's model (the same whatever library is open).
    pub fn library_changed(&mut self) {
        let lane = self.lane.library_changed();
        let describe = self.describe.library_changed();
        *self = State { lane, describe, ..State::default() };
    }
}

/// Whether a photo can be fingerprinted: in the library, and a still.
pub(crate) fn eligible(p: &Photo) -> bool {
    p.in_library() && p.kind != MediaKind::Video
}

pub(crate) fn library_dir(s: &Session) -> Option<PathBuf> {
    s.library.as_ref().filter(|l| l.on_disk).map(|l| l.dir.clone())
}

/// Open the sidecar files of the library in use, once per library: the digests (dropping records of photos no longer
/// in the catalog) and the ignored pairs.
pub(crate) fn ensure_open(s: &mut Session) {
    let dir = library_dir(s);
    if s.vera.opened.as_ref() == Some(&dir) {
        return;
    }
    let digests_path = dir.as_ref().map(|d| d.join(digests::FILE_NAME));
    let embeddings_path = dir.as_ref().map(|d| d.join(embeddings::file_name(lane::MODEL_ID)));
    let digests_moved = s.vera.digests.path() != digests_path.as_deref();
    let embeddings_moved = s.vera.lane.store.path() != embeddings_path.as_deref();
    if digests_moved || embeddings_moved {
        let live: HashSet<String> = s.catalog.photos().map(|p| content_key(p)).collect();
        if digests_moved {
            s.vera.digests.reset(digests_path, Some(&live));
            s.vera.skipped.clear();
            s.vera.queue.clear();
            s.vera.queue_stamp = None;
        }
        if embeddings_moved {
            lane::open_store(s, dir.as_deref(), &live);
        }
    }
    let ignored_path = dir.as_ref().map(|d| d.join(IGNORED_FILE));
    if s.vera.ignored.path() != ignored_path.as_deref() {
        s.vera.ignored = IgnoredPairs::load(ignored_path);
    }
    describe::open(s, dir.as_deref());
    s.vera.opened = Some(dir);
}

/// Both digests of a decoded picture, from its Rec. 601 luma; `None` for a picture too small to fingerprint.
pub(crate) fn digest_image(img: &Rgb32f) -> Option<Digest> {
    let luma = lightcraft_vera::digest::luma_from_rgb(img.width, img.height, &img.data);
    let plane = Luma::new(img.width, img.height, &luma)?;
    lightcraft_vera::digest(&plane)
}

/// Digest photo `id` from its thumbnail-level source. The source is read without being put in the session's source
/// cache: a pass over the whole library would push out the sources the user is working with.
pub(crate) fn digest_photo(s: &mut Session, id: PhotoId) -> Result<Digest, String> {
    let Some(p) = s.catalog.photo(id) else { return Err("no such photo".into()) };
    let source = s.media.source_ref(p, SourceLevel::Thumb);
    let img = source.load()?;
    digest_image(&img).ok_or_else(|| "too small to fingerprint".into())
}

/// Digest photo `id` now unless its picture already has a digest; remembered under its content key. A picture whose
/// source fails is remembered as skipped (the error is returned).
pub(crate) fn digest_now(s: &mut Session, id: PhotoId) -> Result<Option<Digest>, String> {
    let key = s.catalog.photo(id).map(|p| content_key(p)).ok_or("no such photo")?;
    if let Some(d) = s.vera.digests.get(&key) {
        return Ok(Some(*d));
    }
    match digest_photo(s, id) {
        Ok(d) => {
            s.vera.digests.insert(key, d);
            Ok(Some(d))
        }
        Err(e) => {
            s.vera.skipped.insert(key);
            Err(e)
        }
    }
}

/// Look through the catalog for pictures without a digest (newest photos first).
fn refresh_queue(s: &mut Session) {
    let mut by_key: HashMap<String, Vec<PhotoId>> = HashMap::new();
    let (mut photos, mut digested) = (0usize, 0usize);
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        photos += 1;
        let key = content_key(p);
        if s.vera.digests.contains(&key) {
            digested += 1;
        } else if !s.vera.skipped.contains(&key) {
            by_key.entry(key).or_default().push(p.id);
        }
    }
    let mut queue: Vec<Pending> = by_key
        .into_iter()
        .map(|(key, mut ids)| {
            ids.sort();
            Pending { key, ids }
        })
        .collect();
    // taken from the end: the newest photo last
    queue.sort_by_key(|p| p.ids.first().copied());
    s.vera.pending_known = queue.iter().map(|p| p.ids.len()).sum();
    s.vera.photos_known = photos;
    s.vera.digested_known = digested;
    s.vera.queue = queue;
    s.vera.queue_stamp = Some(s.catalog.revision);
    s.vera.queued_at = Some(Instant::now());
}

/// The pump's row in the activity stack: there while photos are pending, its whole the most that were pending at once.
fn track(s: &mut Session, pending: usize) {
    if pending == 0 {
        s.vera.task = None;
        s.vera.peak = 0;
        return;
    }
    s.vera.peak = s.vera.peak.max(pending);
    let task = s.vera.task.get_or_insert_with(|| s.activity.start("vera", "Fingerprinting photos", Cancel::No));
    task.progress(s.vera.peak.saturating_sub(pending) as u64, s.vera.peak as u64);
}

/// Write new digests to the cache file every few seconds, and when the work has run out.
fn save_if_due(s: &mut Session, pending: usize) {
    if s.vera.digests.has_unsaved() && (pending == 0 || s.vera.saved_at.is_none_or(|t| t.elapsed() >= SAVE_EVERY)) {
        if let Err(e) = s.vera.digests.save() {
            log::warn!("vera: could not save the picture digests: {e}");
        }
        s.vera.saved_at = Some(Instant::now());
    }
}

/// One call of `vera.pump`: digest up to the pace's number of pictures, keep the activity row and the cache file up to
/// date, and say how far the library is.
pub(crate) fn pump(s: &mut Session, pace: Pace) -> Value {
    ensure_open(s);
    let per_call = pace.per_call();
    if per_call > 0 {
        let stale = s.vera.queue_stamp != Some(s.catalog.revision);
        if s.vera.queue.is_empty() && stale && s.vera.queued_at.is_none_or(|t| t.elapsed() >= REQUEUE_EVERY) {
            refresh_queue(s);
        }
        let mut tried = 0;
        while tried < per_call {
            let Some(next) = s.vera.queue.pop() else { break };
            s.vera.pending_known = s.vera.pending_known.saturating_sub(next.ids.len());
            // digested since the look (`vera.digest`, `photo.duplicates`), or failed: nothing to do
            if s.vera.digests.contains(&next.key) {
                s.vera.digested_known = s.vera.digested_known.saturating_add(next.ids.len());
                continue;
            }
            if s.vera.skipped.contains(&next.key) {
                continue;
            }
            tried += 1;
            let mut done = None;
            for id in &next.ids {
                match digest_photo(s, *id) {
                    Ok(d) => {
                        done = Some(d);
                        break;
                    }
                    Err(e) => log::debug!("vera: photo {}: {e}", id.0),
                }
            }
            match done {
                Some(d) => {
                    s.vera.digests.insert(next.key, d);
                    s.vera.digested_known = s.vera.digested_known.saturating_add(next.ids.len());
                }
                None => {
                    s.vera.skipped.insert(next.key);
                }
            }
        }
        track(s, s.vera.pending_known);
    } else if s.vera.pending_known == 0 {
        track(s, 0);
    }
    let pending = s.vera.pending_known;
    save_if_due(s, pending);
    let embedding = lane::pump(s, pace);
    let embedding_active = embedding["pending"].as_u64().is_some_and(|n| n > 0);
    // describing runs whatever the pace once asked for (pause only holds back new requests)
    let describe = describe::pump(s, pace);
    let describe_active = describe["pending"].as_u64().is_some_and(|n| n > 0) || describe["inFlight"].as_u64().is_some_and(|n| n > 0);
    json!({
        "active": pending > 0 || embedding_active || describe_active,
        "pending": pending,
        "digested": s.vera.digested_known,
        "photos": s.vera.photos_known,
        "pace": pace.name(),
        "embedding": embedding,
        "describe": describe,
    })
}

/// The keyword lane's fields of a photo, in [`FIELD_WEIGHTS`] order. Hierarchical keywords (`travel|Italy|Rome`)
/// contribute every level.
pub(crate) fn fields(p: &Photo) -> [String; 7] {
    let m = &p.meta;
    let keywords = m.keywords.iter().map(|k| k.replace('|', " ")).collect::<Vec<_>>().join(" ");
    let stem = Path::new(&p.file_name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let gear = [m.camera.as_str(), &m.lens, &m.location, &m.city, &m.state, &m.country].join(" ");
    [m.title.clone(), keywords, m.caption.clone(), m.alt_text.clone(), m.extended_description.clone(), stem, gear]
}

/// The keyword index, built from the catalog (every photo in the library) when there is none or the catalog changed
/// since it was built. A full rebuild: cheap enough at library sizes (`docs/vera.md`), and logged so it can be watched.
pub(crate) fn ensure_index(s: &mut Session) {
    if s.vera.index.is_some() && s.vera.index_rev == s.catalog.revision {
        return;
    }
    let started = Instant::now();
    let mut index = Index::new(&FIELD_WEIGHTS);
    let mut n = 0usize;
    for p in s.catalog.photos().filter(|p| p.in_library()) {
        let f = fields(p);
        let refs: [&str; 7] = f.each_ref().map(String::as_str);
        index.upsert(p.id.0, &refs);
        n += 1;
    }
    log::debug!("vera: keyword index of {n} photos built in {} ms", started.elapsed().as_millis());
    s.vera.index = Some(index);
    s.vera.index_rev = s.catalog.revision;
}

/// What the import review's same-picture check (PRD C5) needs from the session, so it can run on the scan's thread:
/// the library's digests, the ignored pairs, the file loader and the candidate digests made by earlier reviews.
pub(crate) struct NearCheck {
    /// One entry per picture: a photo that shows it, its content key, its digest.
    library: Vec<(PhotoId, String, Digest)>,
    ignored: IgnoredPairs,
    loader: Option<crate::media::FileLoader>,
    /// Candidate digests by (path, file size); what this check adds is handed back ([`NearCheck::fresh`]).
    cache: HashMap<(String, u64), Digest>,
    fresh: Vec<((String, u64), Digest)>,
}

/// The import check for the library in use, or `None` while the library has no digests (nothing would match).
pub(crate) fn near_check(s: &mut Session) -> Option<NearCheck> {
    ensure_open(s);
    if s.vera.digests.is_empty() {
        return None;
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut library: Vec<(PhotoId, String, Digest)> = Vec::new();
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        let key = content_key(p);
        if let Some(d) = s.vera.digests.get(&key).copied()
            && seen.insert(key.clone())
        {
            library.push((p.id, key, d));
        }
    }
    if library.is_empty() {
        return None;
    }
    Some(NearCheck {
        library,
        ignored: s.vera.ignored.clone(),
        loader: s.media.file_loader.clone(),
        cache: s.vera.candidates.clone(),
        fresh: Vec::new(),
    })
}

impl NearCheck {
    /// Check the candidates: each one that is not already a duplicate by path or bytes (nor unreadable, nor a video) is
    /// decoded at thumbnail level, digested and compared with every digest in the library (brute force; at most
    /// [`MAX_IMPORT_CHECKS`] candidates). The first same picture wins, else the closest possible one. A match is set on
    /// the candidate and on its probe, which carries it to the import that follows. Stops when `cancel` is set.
    pub fn run(&mut self, candidates: &mut [ImportCandidate], probes: &mut HashMap<String, crate::media::ProbeInfo>, cancel: &AtomicBool) {
        let mut checked = 0usize;
        for c in candidates.iter_mut() {
            if c.duplicate.is_some() || c.error.is_some() || c.kind == MediaKind::Video {
                continue;
            }
            if checked >= MAX_IMPORT_CHECKS || cancel.load(Ordering::Relaxed) {
                break;
            }
            checked += 1;
            let Some(digest) = self.digest_of(&c.path, c.file_size) else { continue };
            // the key the file will have once imported, for pairs the user said are not the same
            let own = probes.get(&c.path).and_then(|i| i.content_hash.clone());
            let Some(nd) = self.best_match(&digest, own.as_deref()) else { continue };
            if let Some(info) = probes.get_mut(&c.path) {
                info.near_duplicate = Some(nd.clone());
            }
            c.near_duplicate = Some(nd);
        }
    }

    fn digest_of(&mut self, path: &str, size: u64) -> Option<Digest> {
        let cache_key = (path.to_string(), size);
        if let Some(d) = self.cache.get(&cache_key) {
            return Some(*d);
        }
        let loader = self.loader.as_ref()?;
        let d = match loader(path, SourceLevel::Thumb.max_edge()) {
            Ok((img, _)) => digest_image(&img)?,
            Err(e) => {
                log::debug!("vera: import check {path}: {e}");
                return None;
            }
        };
        self.cache.insert(cache_key.clone(), d);
        self.fresh.push((cache_key, d));
        Some(d)
    }

    fn best_match(&self, digest: &Digest, own: Option<&str>) -> Option<NearDuplicate> {
        let mut best: Option<(PhotoId, &'static str, u32, u32)> = None;
        for (id, key, d) in &self.library {
            if own.is_some_and(|h| self.ignored.contains(h, key)) {
                continue;
            }
            let (pdq, phash) = lightcraft_vera::digest::distances(digest, d);
            match lightcraft_vera::sameness(digest, d) {
                Sameness::Same => {
                    best = Some((*id, "picture", pdq, phash));
                    break;
                }
                Sameness::Possible if best.is_none_or(|b| (pdq, phash) < (b.2, b.3)) => best = Some((*id, "possible", pdq, phash)),
                _ => {}
            }
        }
        best.map(|(existing, tier, pdq, phash)| NearDuplicate { existing: existing.0, tier: tier.to_string(), pdq, phash })
    }

    /// The candidate digests this check made, for the session's cache ([`remember_candidates`]).
    pub fn fresh(self) -> Vec<((String, u64), Digest)> {
        self.fresh
    }
}

/// Keep candidate digests a review made, so a second review of the same files does not decode them again.
pub(crate) fn remember_candidates(s: &mut Session, fresh: Vec<((String, u64), Digest)>) {
    if s.vera.candidates.len().saturating_add(fresh.len()) > MAX_CANDIDATE_CACHE {
        s.vera.candidates.clear();
    }
    s.vera.candidates.extend(fresh);
}

/// Pictures the user said are not the same, as ordered pairs of content keys (so `(a, b)` and `(b, a)` are one pair).
#[derive(Clone, Default)]
pub(crate) struct IgnoredPairs {
    set: BTreeSet<(String, String)>,
    path: Option<PathBuf>,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct IgnoredFile {
    version: u32,
    pairs: Vec<(String, String)>,
}

fn ordered(a: &str, b: &str) -> (String, String) {
    if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) }
}

fn valid_key(k: &str) -> bool {
    !k.is_empty() && k.len() <= MAX_PAIR_KEY
}

impl IgnoredPairs {
    /// The pairs in the file at `path` (none when it is missing, too large or not ours).
    pub fn load(path: Option<PathBuf>) -> IgnoredPairs {
        let mut set = BTreeSet::new();
        if let Some(p) = &path
            && let Some(bytes) = crate::denoise::read_capped(p, MAX_IGNORED_FILE)
        {
            match serde_json::from_slice::<IgnoredFile>(&bytes) {
                Ok(f) => {
                    for (a, b) in f.pairs.iter().filter(|(a, b)| valid_key(a) && valid_key(b)).take(MAX_IGNORED) {
                        set.insert(ordered(a, b));
                    }
                }
                Err(e) => log::warn!("vera: {} is unreadable ({e}); starting without ignored pairs", p.display()),
            }
        }
        IgnoredPairs { set, path }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn contains(&self, a: &str, b: &str) -> bool {
        !self.set.is_empty() && self.set.contains(&ordered(a, b))
    }

    /// Remember a pair; `Ok(false)` when it was already there.
    pub fn insert(&mut self, a: &str, b: &str) -> Result<bool, String> {
        if !valid_key(a) || !valid_key(b) {
            return Err("these photos have no key to remember them by".into());
        }
        if self.set.len() >= MAX_IGNORED && !self.contains(a, b) {
            return Err(format!("at most {MAX_IGNORED} pairs can be kept apart"));
        }
        Ok(self.set.insert(ordered(a, b)))
    }

    /// Forget a pair; `false` when it was not there.
    pub fn remove(&mut self, a: &str, b: &str) -> bool {
        self.set.remove(&ordered(a, b))
    }

    /// Write the pairs (atomically) beside the library; an in-memory library keeps them for the session.
    pub fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else { return Ok(()) };
        let file = IgnoredFile { version: 1, pairs: self.set.iter().cloned().collect() };
        let bytes = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
        crate::denoise::write_atomic(path, &bytes)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_describe;
#[cfg(test)]
mod tests_lane;
