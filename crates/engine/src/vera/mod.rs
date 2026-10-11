//! VERA in the engine: *find by words* (the keyword lane) and *same picture* (the identity lane) over the library.
//!
//! The algorithms live in `lightcraft-vera` (pure functions, no I/O); this module decides what to digest and when, where
//! the results are kept and how the commands see them (`docs/vera.md`):
//! - [`digests`]: one fingerprint per picture (PDQ + pHash over the CPNP-1 Canonical Buffer, with the VER `pixel_hash`
//!   and `content_hash`; provisional thumbnail-level digests for a file without a CPNP-1 path), keyed by content,
//!   cached in `vera-digests.bin` beside the library; made off the session thread by [`fingerprint`].
//! - [`sidecar`]: where that file and the others below live: the library folder on disk, or the library's own file
//!   store (the browser's storage); memory only without a library.
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
pub(crate) mod fingerprint;
pub(crate) mod keywords;
pub(crate) mod lane;
pub(crate) mod model;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod remote;
pub(crate) mod review;
pub(crate) mod sidecar;
pub(crate) mod vocabulary;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod worker;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lightcraft_catalog::{MediaKind, Photo, PhotoId};
use lightcraft_raster::Rgb32f;
use lightcraft_vera::{Digest, Index, Sameness};
use serde_json::{Value, json};
use web_time::Instant;

use crate::Session;
use crate::activity::{Cancel, TaskGuard};
use crate::import::{ImportCandidate, NearDuplicate};
use crate::media::{SourceLevel, content_key};
use digests::Fingerprint;
use sidecar::{Loaded, Sidecars};

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
/// On the web the pump's digest step decodes on the main thread (no threads there): it stops starting pictures once it
/// has worked this long in one call, so a few large raws never stall a frame (a decode already started still finishes).
pub(crate) const WEB_STEP_BUDGET: Duration = Duration::from_millis(8);

/// The digest step's time budget per call: [`WEB_STEP_BUDGET`] in the browser, none on the desktop (the pace's
/// picture count alone bounds it there).
pub(crate) fn step_budget() -> Option<Duration> {
    cfg!(target_arch = "wasm32").then_some(WEB_STEP_BUDGET)
}

/// How much of one call's digest step is used: pictures started, and the time since the step began.
pub(crate) struct Step {
    started: Instant,
    tried: usize,
    per_call: usize,
    budget: Option<Duration>,
}

impl Step {
    pub fn new(per_call: usize, budget: Option<Duration>, started: Instant) -> Step {
        Step { started, tried: 0, per_call, budget }
    }

    /// Whether another picture may be started at `now`: under the pace's count, and within the time budget. The first
    /// picture always may (a budget bounds the work after it, not whether there is any).
    pub fn may_start(&self, now: Instant) -> bool {
        self.tried < self.per_call && (self.tried == 0 || self.budget.is_none_or(|b| now.saturating_duration_since(self.started) < b))
    }

    pub fn started_one(&mut self) {
        self.tried += 1;
    }
}

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
    pub(crate) fn per_call(self) -> usize {
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
    /// Where the sidecar files were opened from (`None`: not yet for this library).
    opened: Option<Sidecars>,
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
    /// The fingerprinting thread (started on first need) and the pictures handed to it, with how many photos show each.
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<fingerprint::Worker>,
    in_flight: HashMap<String, usize>,
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

/// Open the sidecar files of the library in use, once per library: the digests (dropping records of photos no longer
/// in the catalog), the embeddings, the ignored pairs and the review queue, from the library folder or the library's
/// file store ([`sidecar::of`]).
pub(crate) fn ensure_open(s: &mut Session) {
    let files = sidecar::of(s);
    if s.vera.opened.as_ref().is_some_and(|o| o.same_place(&files)) {
        return;
    }
    let digests_moved = !s.vera.digests.files().same_place(&files);
    let embeddings_moved = !s.vera.lane.store.files().same_place(&files);
    if digests_moved || embeddings_moved {
        let live: HashSet<String> = s.catalog.photos().map(|p| content_key(p)).collect();
        if digests_moved {
            s.vera.digests.reset(files.clone(), Some(&live));
            s.vera.skipped.clear();
            s.vera.queue.clear();
            s.vera.queue_stamp = None;
        }
        if embeddings_moved {
            lane::open_store(s, &files, &live);
        }
    }
    if !s.vera.ignored.files().same_place(&files) {
        s.vera.ignored = IgnoredPairs::load(files.clone());
    }
    describe::open(s, &files);
    s.vera.opened = Some(files);
}

/// Provisional digests of a decoded picture ([`fingerprint::digest_linear`]); `None` for a picture too small to
/// fingerprint.
pub(crate) fn digest_image(img: &Rgb32f) -> Option<Digest> {
    fingerprint::digest_linear(img)
}

/// What photo `id`'s picture is fingerprinted from: its file, and its thumbnail-level source (read without being put in
/// the session's source cache: a pass over the whole library would push out the sources the user is working with).
fn input_of(s: &mut Session, id: PhotoId) -> Option<fingerprint::Input> {
    let p = s.catalog.photo(id)?.clone();
    let path = match &p.source {
        lightcraft_catalog::Source::File { path } => Some(path.clone()),
        lightcraft_catalog::Source::Demo { .. } => None,
    };
    Some(fingerprint::Input { path, read: s.media.file_bytes.clone(), thumb: s.media.source_ref(&p, SourceLevel::Thumb) })
}

/// Fingerprint photo `id` now, on this thread ([`fingerprint::make`]).
pub(crate) fn digest_photo(s: &mut Session, id: PhotoId) -> Result<Fingerprint, String> {
    let input = input_of(s, id).ok_or("no such photo")?;
    fingerprint::make(&input)
}

/// Fingerprint photo `id` now unless its picture already has one; remembered under its content key. A picture whose
/// source fails is remembered as skipped (the error is returned).
pub(crate) fn fingerprint_now(s: &mut Session, id: PhotoId) -> Result<Option<Fingerprint>, String> {
    let key = s.catalog.photo(id).map(|p| content_key(p)).ok_or("no such photo")?;
    if let Some(f) = s.vera.digests.fingerprint(&key) {
        return Ok(Some(*f));
    }
    match digest_photo(s, id) {
        Ok(f) => {
            s.vera.digests.insert_fingerprint(key, f);
            Ok(Some(f))
        }
        Err(e) => {
            s.vera.skipped.insert(key);
            Err(e)
        }
    }
}

/// [`fingerprint_now`]'s digests.
pub(crate) fn digest_now(s: &mut Session, id: PhotoId) -> Result<Option<Digest>, String> {
    fingerprint_now(s, id).map(|f| f.map(|f| f.digest))
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
        } else if !s.vera.skipped.contains(&key) && !s.vera.in_flight.contains_key(&key) {
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

/// Take in what the fingerprinting thread has finished.
#[cfg(not(target_arch = "wasm32"))]
fn collect(s: &mut Session) {
    let finished = s.vera.worker.as_ref().map(fingerprint::Worker::finished).unwrap_or_default();
    for d in finished {
        s.vera.in_flight.remove(&d.key);
        match d.result {
            Ok(f) => {
                s.vera.digests.insert_fingerprint(d.key, f);
                s.vera.digested_known = s.vera.digested_known.saturating_add(d.photos);
            }
            Err(e) => {
                log::debug!("vera: picture {}: {e}", d.key);
                s.vera.skipped.insert(d.key);
            }
        }
    }
}

/// Fingerprint one pending picture: on the worker when it has a file (a full-size decode), here otherwise (a demo
/// scene, the browser). `false` when the worker could not take it (it is queued again).
fn start(s: &mut Session, next: Pending) -> bool {
    let inputs: Vec<(PhotoId, fingerprint::Input)> = next.ids.iter().filter_map(|id| input_of(s, *id).map(|i| (*id, i))).collect();
    #[cfg(not(target_arch = "wasm32"))]
    if inputs.iter().any(|(_, i)| i.path.is_some()) {
        if s.vera.worker.is_none() {
            s.vera.worker = fingerprint::Worker::start();
        }
        let job = fingerprint::Job { key: next.key.clone(), photos: next.ids.len(), inputs };
        if s.vera.worker.as_ref().is_some_and(|w| w.submit(job)) {
            s.vera.in_flight.insert(next.key, next.ids.len());
            return true;
        }
        // the thread is gone: start another on the next call
        s.vera.worker = None;
        s.vera.queue.push(next);
        return false;
    }
    let mut done = None;
    for (id, input) in &inputs {
        match fingerprint::make(input) {
            Ok(f) => {
                done = Some(f);
                break;
            }
            Err(e) => log::debug!("vera: photo {}: {e}", id.0),
        }
    }
    match done {
        Some(f) => {
            s.vera.digests.insert_fingerprint(next.key, f);
            s.vera.digested_known = s.vera.digested_known.saturating_add(next.ids.len());
        }
        None => {
            s.vera.skipped.insert(next.key);
        }
    }
    true
}

/// One call of `vera.pump`: fingerprint up to the pace's number of pictures, keep the activity row and the cache file
/// up to date, and say how far the library is.
pub(crate) fn pump(s: &mut Session, pace: Pace) -> Value {
    ensure_open(s);
    #[cfg(not(target_arch = "wasm32"))]
    collect(s);
    let per_call = pace.per_call();
    if per_call > 0 {
        let stale = s.vera.queue_stamp != Some(s.catalog.revision);
        if s.vera.queue.is_empty() && s.vera.in_flight.is_empty() && stale && s.vera.queued_at.is_none_or(|t| t.elapsed() >= REQUEUE_EVERY) {
            refresh_queue(s);
        }
        let mut step = Step::new(per_call, step_budget(), Instant::now());
        while step.may_start(Instant::now()) && s.vera.in_flight.len() < fingerprint::MAX_IN_FLIGHT {
            let Some(next) = s.vera.queue.pop() else { break };
            s.vera.pending_known = s.vera.pending_known.saturating_sub(next.ids.len());
            // fingerprinted since the look (`vera.digest`, `photo.duplicates`), or failed: nothing to do
            if s.vera.digests.contains(&next.key) {
                s.vera.digested_known = s.vera.digested_known.saturating_add(next.ids.len());
                continue;
            }
            if s.vera.skipped.contains(&next.key) || s.vera.in_flight.contains_key(&next.key) {
                continue;
            }
            step.started_one();
            let n = next.ids.len();
            if !start(s, next) {
                s.vera.pending_known = s.vera.pending_known.saturating_add(n);
                break;
            }
        }
    }
    let pending = s.vera.pending_known.saturating_add(s.vera.in_flight.values().sum::<usize>());
    if per_call > 0 || pending == 0 {
        track(s, pending);
    }
    save_if_due(s, pending);
    let embedding = lane::pump(s, pace);
    let embedding_active = embedding["pending"].as_u64().is_some_and(|n| n > 0);
    // describing runs whatever the pace once asked for (pause only holds back new requests)
    let describe = describe::pump(s, pace);
    let describe_active = describe["pending"].as_u64().is_some_and(|n| n > 0) || describe["inFlight"].as_u64().is_some_and(|n| n > 0);
    let ver = crate::ver::export::pump(s, pace);
    let ver_active = ver["pending"].as_u64().is_some_and(|n| n > 0);
    json!({
        "active": pending > 0 || embedding_active || describe_active || ver_active,
        "pending": pending,
        "digested": s.vera.digested_known,
        "photos": s.vera.photos_known,
        "pace": pace.name(),
        "embedding": embedding,
        "describe": describe,
        "ver": ver,
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
    files: Sidecars,
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
    /// The pairs in the file kept in `files` (none when it is missing, too large or not ours).
    pub fn load(files: Sidecars) -> IgnoredPairs {
        let mut set = BTreeSet::new();
        if let Loaded::Bytes(bytes) = files.read(IGNORED_FILE, MAX_IGNORED_FILE as u64) {
            match serde_json::from_slice::<IgnoredFile>(&bytes) {
                Ok(f) => {
                    for (a, b) in f.pairs.iter().filter(|(a, b)| valid_key(a) && valid_key(b)).take(MAX_IGNORED) {
                        set.insert(ordered(a, b));
                    }
                }
                Err(e) => log::warn!("vera: {} is unreadable ({e}); starting without ignored pairs", files.describe(IGNORED_FILE)),
            }
        }
        IgnoredPairs { set, files }
    }

    /// Where the file is kept.
    pub fn files(&self) -> &Sidecars {
        &self.files
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

    /// Write the pairs (atomically) beside the library, or into its file store; without a library they are kept for the
    /// session.
    pub fn save(&self) -> Result<(), String> {
        if !self.files.persists() {
            return Ok(());
        }
        let file = IgnoredFile { version: 1, pairs: self.set.iter().cloned().collect() };
        let bytes = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
        self.files.write_atomic(IGNORED_FILE, &bytes)
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_describe;
#[cfg(test)]
mod tests_lane;
#[cfg(test)]
mod tests_sidecar;
