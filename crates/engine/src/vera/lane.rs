//! The vector lane in the session: the model (SigLIP 2, downloaded on request), the embeddings of the library's
//! pictures, the background pump that makes them and the query side that reads them (`docs/vera.md` → Phase 2).
//!
//! - [`Lane::store`]: one canonical vector per picture ([`super::embeddings`]), beside the library.
//! - [`Engine`]: the loaded model, the embed worker ([`super::worker`]), the download and the query cache. It outlives
//!   a library change (the model is the same whatever library is open); everything else is per library.
//! - [`pump`]: called by `vera.pump` every frame; takes in finished vectors, prepares at most the pace's number of
//!   render jobs for pictures without one (newest first) and keeps the "Embedding photos" activity row.
//!
//! Nothing here runs on the web build: the lane reports `needs_desktop` there.

// the web build runs none of the lane: what only the desktop uses is unused there
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_variables, unused_mut))]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::time::Duration;

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};
use web_time::Instant;

use super::embeddings::EmbeddingStore;
use super::{Pace, eligible};
use crate::Session;
use crate::activity::{Cancel, TaskGuard};
use crate::media::content_key;

/// The model the lane uses, as the sidecar file and `vera.status` name it.
pub(crate) const MODEL_ID: &str = lightcraft_embed::MODEL_ID;
/// The user's own download locations for the model, one base URL per line, beside the model folder.
pub(crate) const MIRRORS_FILE: &str = "siglip2-mirrors.txt";
/// Resident memory of the loaded model with its image tower (measured: about 380 MB of weights plus the runtime).
pub(crate) const MODEL_BYTES: u64 = 420 << 20;
/// One batch of model inputs in flight (`MAX_BATCH` pictures of 256 × 256 × 3 f32).
pub(crate) const BATCH_BYTES: u64 = (16 * 256 * 256 * 3 * 4) as u64;
/// Most pictures handed to the worker and not back yet: two batches, so the worker always has the next one.
const MAX_IN_FLIGHT: usize = 32;
/// How often the pump writes new vectors to the cache file (and whenever the work runs out).
const SAVE_EVERY: Duration = Duration::from_secs(3);
/// How often the pump may look through the catalog for pictures without a vector.
const REQUEUE_EVERY: Duration = Duration::from_secs(1);
/// After the model failed, how long before it is tried again.
const RETRY_AFTER: Duration = Duration::from_secs(30);
/// Query embeddings remembered (the cache is emptied when full).
const MAX_QUERIES: usize = 256;

/// Pictures still to embed: one content key, and the photos that show it (the first that renders is used).
struct Pending {
    key: String,
    ids: Vec<PhotoId>,
}

/// What lasts across libraries: the model, its worker, the download and the query cache.
#[derive(Default)]
pub(crate) struct Engine {
    /// The model folder it was loaded from, and the model.
    #[cfg(not(target_arch = "wasm32"))]
    model: Option<(PathBuf, Arc<lightcraft_embed::Model>)>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<super::worker::Worker>,
    #[cfg(not(target_arch = "wasm32"))]
    pub download: crate::segment::download::Downloader,
    /// Why the model cannot be used (it failed to load or to run), until [`Engine::retry_at`].
    pub error: Option<String>,
    retry_at: Option<Instant>,
    /// Query → its text embedding.
    queries: HashMap<String, Vec<f32>>,
    /// Bumped when another library is opened: what the worker finishes for the old one is dropped.
    epoch: u64,
}

impl Engine {
    /// Let go of the model and its worker (their memory is freed once the worker has finished what it holds).
    pub fn unload(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.worker = None;
            self.model = None;
        }
        self.queries.clear();
    }

    pub fn loaded(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.model.is_some();
        #[cfg(target_arch = "wasm32")]
        false
    }
}

/// The vector lane's part of [`super::State`].
pub(crate) struct Lane {
    pub store: EmbeddingStore,
    /// The user's choice (`vera.config`, saved in the library's prefs); `None`: on, the default once the model is
    /// installed.
    pub vector_lane: Option<bool>,
    /// Pictures that could not be rendered this session (not tried again every frame).
    pub skipped: HashSet<String>,
    pub engine: Engine,
    queue: Vec<Pending>,
    queue_stamp: Option<u64>,
    queued_at: Option<Instant>,
    /// Pictures handed to the worker, with how many photos show each.
    in_flight: HashMap<String, usize>,
    task: Option<TaskGuard>,
    saved_at: Option<Instant>,
    peak: usize,
    /// Photos waiting in the queue, photos that can be embedded and photos that are, as of the last look (kept up to
    /// date as the pump works).
    pending_known: usize,
    photos_known: usize,
    embedded_known: usize,
    /// Query embeddings a test provides, used instead of the model's (the fusion tests need no checkpoint).
    #[cfg(test)]
    pub test_queries: HashMap<String, Vec<f32>>,
}

impl Default for Lane {
    fn default() -> Self {
        Lane {
            store: EmbeddingStore::new(MODEL_ID, lightcraft_embed::DIM),
            vector_lane: None,
            skipped: HashSet::new(),
            engine: Engine::default(),
            queue: Vec::new(),
            queue_stamp: None,
            queued_at: None,
            in_flight: HashMap::new(),
            task: None,
            saved_at: None,
            peak: 0,
            pending_known: 0,
            photos_known: 0,
            embedded_known: 0,
            #[cfg(test)]
            test_queries: HashMap::new(),
        }
    }
}

impl Lane {
    /// Another library was opened: its vectors are read afresh; the model, the worker and the download stay.
    pub fn library_changed(&mut self) -> Lane {
        let mut engine = std::mem::take(&mut self.engine);
        engine.epoch = engine.epoch.wrapping_add(1);
        Lane {
            engine,
            #[cfg(test)]
            test_queries: std::mem::take(&mut self.test_queries),
            ..Lane::default()
        }
    }

    /// Whether the user wants the lane (on unless switched off).
    pub fn wanted(&self) -> bool {
        self.vector_lane.unwrap_or(true)
    }

    /// Let go of the model and its worker, and of what the worker held (looked for again by the next pump).
    pub fn unload(&mut self) {
        self.engine.unload();
        self.in_flight.clear();
        self.requeue();
    }

    /// Forget the queue so the next pump looks at the catalog again.
    pub fn requeue(&mut self) {
        self.queue.clear();
        self.queue_stamp = None;
        self.queued_at = None;
    }
}

/// Where the vector lane stands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LaneState {
    Live,
    /// No model in the model folder (`dir`; `None` when the build has nowhere to keep one).
    NeedsModel(Option<PathBuf>),
    Downloading {
        done: u64,
        total: u64,
    },
    DisabledByUser,
    NeedsDesktop,
    Error(String),
}

impl LaneState {
    pub fn name(&self) -> &'static str {
        match self {
            LaneState::Live => "live",
            LaneState::NeedsModel(_) => "needs_model",
            LaneState::Downloading { .. } => "downloading",
            LaneState::DisabledByUser => "disabled_by_user",
            LaneState::NeedsDesktop => "needs_desktop",
            LaneState::Error(_) => "error",
        }
    }

    /// The sentence `vera.status` shows with the state.
    pub fn reason(&self) -> String {
        let gb = download_bytes() as f64 / 1e9;
        match self {
            LaneState::Live => format!("Finding photos by meaning with SigLIP 2 ({MODEL_ID}), on this computer."),
            LaneState::NeedsModel(Some(dir)) => format!(
                "Finding photos by meaning needs the SigLIP 2 model (about {gb:.1} GB, Apache License 2.0): download it with vera.model.download {{\"acceptLicense\": true}} once the user has agreed to its licence, or put its files in {}.",
                dir.display()
            ),
            LaneState::NeedsModel(None) => "Finding photos by meaning needs the SigLIP 2 model, and this build has nowhere to keep it.".into(),
            LaneState::Downloading { done, total } => {
                let pct = done.saturating_mul(100).checked_div(*total).unwrap_or(0);
                format!("The SigLIP 2 model is downloading ({pct} %); finding photos by meaning starts when it is in place.")
            }
            LaneState::DisabledByUser => {
                "Finding photos by meaning is switched off for this library (vera.config {\"vectorLane\": true} switches it on).".into()
            }
            LaneState::NeedsDesktop => "Searching by meaning needs the desktop app for now: the browser cannot run the model".into(),
            LaneState::Error(e) => format!("The SigLIP 2 model could not be used: {e}"),
        }
    }
}

/// Bytes the model download fetches.
pub(crate) fn download_bytes() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    return lightcraft_embed::fetch::SIGLIP2_FILES.iter().filter_map(|f| f.size).sum();
    #[cfg(target_arch = "wasm32")]
    1_511_714_589
}

/// Whether the model's files are in the model folder.
pub(crate) fn installed(s: &Session) -> bool {
    s.vera_model_dir.as_deref().is_some_and(lightcraft_embed::is_model_dir)
}

pub(crate) fn state(s: &Session) -> LaneState {
    if cfg!(target_arch = "wasm32") {
        return LaneState::NeedsDesktop;
    }
    if !installed(s) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let d = s.vera.lane.engine.download.status();
            if d.running {
                return LaneState::Downloading { done: d.done, total: d.total };
            }
        }
        return LaneState::NeedsModel(s.vera_model_dir.clone());
    }
    if !s.vera.lane.wanted() {
        return LaneState::DisabledByUser;
    }
    if let Some(e) = &s.vera.lane.engine.error {
        return LaneState::Error(e.clone());
    }
    LaneState::Live
}

/// Open the embeddings of the library whose files are in `files` (dropping records of pictures no longer in its
/// catalog, `live`).
pub(crate) fn open_store(s: &mut Session, files: &super::sidecar::Sidecars, live: &HashSet<String>) {
    if !s.vera.lane.store.files().same_place(files) {
        s.vera.lane.store.reset(files.clone(), Some(live));
        s.vera.lane.skipped.clear();
        s.vera.lane.requeue();
    }
}

/// The photos (in the library, stills) that show each picture, by content key; ids ascending.
pub(crate) fn photos_by_key(s: &Session) -> HashMap<String, Vec<PhotoId>> {
    let mut by: HashMap<String, Vec<PhotoId>> = HashMap::new();
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        by.entry(content_key(p)).or_default().push(p.id);
    }
    for ids in by.values_mut() {
        ids.sort();
    }
    by
}

/// (photos that can be embedded, embedded, waiting: not embedded and not skipped).
pub(crate) fn counts(s: &Session) -> (usize, usize, usize) {
    let (mut photos, mut embedded, mut skipped) = (0usize, 0usize, 0usize);
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        photos += 1;
        let key = content_key(p);
        if s.vera.lane.store.contains(&key) {
            embedded += 1;
        } else if s.vera.lane.skipped.contains(&key) {
            skipped += 1;
        }
    }
    (photos, embedded, photos.saturating_sub(embedded).saturating_sub(skipped))
}

/// Pictures the lane could not render this session, by photo.
pub(crate) fn unreadable(s: &Session) -> usize {
    if s.vera.lane.skipped.is_empty() {
        return 0;
    }
    s.catalog.photos().filter(|p| eligible(p) && s.vera.lane.skipped.contains(&content_key(p))).count()
}

/// The model, loaded from the model folder when it is not yet (cheap: the weights are read when a tower is first
/// used). An error says why the lane cannot run.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn model(s: &mut Session) -> Result<Arc<lightcraft_embed::Model>, String> {
    let st = state(s);
    let lane = &mut s.vera.lane;
    if let LaneState::Error(e) = &st {
        if lane.engine.retry_at.is_some_and(|t| Instant::now() < t) {
            return Err(st.reason());
        }
        log::info!("vera: trying the model again after: {e}");
        lane.engine.error = None;
        lane.unload();
    } else if st != LaneState::Live {
        return Err(st.reason());
    }
    let dir = s.vera_model_dir.clone().ok_or_else(|| LaneState::NeedsModel(None).reason())?;
    let lane = &mut s.vera.lane;
    if let Some((d, m)) = &lane.engine.model
        && *d == dir
    {
        return Ok(m.clone());
    }
    lane.unload();
    match lightcraft_embed::Model::load(&dir, lightcraft_embed::best_device()) {
        Ok(m) => {
            let m = Arc::new(m);
            log::info!("vera: SigLIP 2 model opened from {}", dir.display());
            lane.engine.model = Some((dir, m.clone()));
            Ok(m)
        }
        Err(e) => Err(fail(s, e.to_string())),
    }
}

/// The model failed: the lane stops with `e` until [`RETRY_AFTER`]; returns the sentence `vera.status` shows.
#[cfg(not(target_arch = "wasm32"))]
fn fail(s: &mut Session, e: String) -> String {
    log::warn!("vera: the SigLIP 2 model could not be used: {e}");
    s.vera.lane.unload();
    let engine = &mut s.vera.lane.engine;
    engine.retry_at = Some(Instant::now() + RETRY_AFTER);
    let st = LaneState::Error(e.clone());
    engine.error = Some(e);
    st.reason()
}

/// The text embedding of `query` for the vector lane: `Ok(None)` when the lane is not running (not installed, switched
/// off, the web build); an error when it should run and could not (the text tower failed to load).
pub(crate) fn query_vector(s: &mut Session, query: &str) -> Result<Option<Vec<f32>>, String> {
    #[cfg(test)]
    if let Some(v) = s.vera.lane.test_queries.get(query) {
        return Ok(Some(v.clone()));
    }
    if !matches!(state(s), LaneState::Live | LaneState::Error(_)) {
        return Ok(None);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let norm = query.trim().to_lowercase();
        if let Some(v) = s.vera.lane.engine.queries.get(&norm) {
            return Ok(Some(v.clone()));
        }
        let m = model(s)?;
        let started = Instant::now();
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.embed_query(&norm)));
        let v = match run {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(fail(s, e.to_string())),
            Err(_) => return Err(fail(s, "the text tower stopped unexpectedly".into())),
        };
        log::debug!("vera: query embedded in {} ms", started.elapsed().as_millis());
        let cache = &mut s.vera.lane.engine.queries;
        if cache.len() >= MAX_QUERIES {
            cache.clear();
        }
        cache.insert(norm, v.clone());
        Ok(Some(v))
    }
    #[cfg(target_arch = "wasm32")]
    Ok(None)
}

/// Photo `id`'s canonical vector: from the store, else (with `embed`, when the lane runs) embedded here and now on the
/// session thread, one picture. `Ok(None)` when there is none and none can be made; an error when making it failed.
pub(crate) fn vector_of(s: &mut Session, id: PhotoId, embed: bool) -> Result<Option<Vec<f32>>, String> {
    let key = s.catalog.photo(id).map(|p| content_key(p)).ok_or("no such photo")?;
    if let Some(v) = s.vera.lane.store.get(&key) {
        return Ok(Some(v.to_vec()));
    }
    if !embed || state(s) != LaneState::Live || s.vera.lane.skipped.contains(&key) {
        return Ok(None);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let m = model(s)?;
        let Some((render, preview)) = super::worker::prepare(s, id) else { return Err("this photo cannot be rendered".into()) };
        let pic = match super::worker::picture(render, preview) {
            Ok(p) => p,
            Err(e) => {
                s.vera.lane.skipped.insert(key);
                return Err(e);
            }
        };
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.embed_image(&pic)));
        let v = match run {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => return Err(fail(s, e.to_string())),
            Err(_) => return Err(fail(s, "the image tower stopped unexpectedly".into())),
        };
        if !s.vera.lane.store.insert(key.clone(), &v) {
            return Err("the model returned an unusable embedding".into());
        }
        Ok(s.vera.lane.store.get(&key).map(<[f32]>::to_vec))
    }
    #[cfg(target_arch = "wasm32")]
    Ok(None)
}

/// Look through the catalog for pictures without a vector (newest photos first).
fn refresh_queue(s: &mut Session) {
    let lane = &s.vera.lane;
    let mut by_key: HashMap<String, Vec<PhotoId>> = HashMap::new();
    let (mut photos, mut embedded) = (0usize, 0usize);
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        photos += 1;
        let key = content_key(p);
        if lane.store.contains(&key) {
            embedded += 1;
        } else if !lane.skipped.contains(&key) && !lane.in_flight.contains_key(&key) {
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
    let lane = &mut s.vera.lane;
    lane.pending_known = queue.iter().map(|p| p.ids.len()).sum();
    lane.photos_known = photos;
    lane.embedded_known = embedded;
    lane.queue = queue;
    lane.queue_stamp = Some(s.catalog.revision);
    lane.queued_at = Some(Instant::now());
}

/// The pump's row in the activity stack: there while photos are pending, its whole the most that were pending at once.
fn track(s: &mut Session, pending: usize) {
    let lane = &mut s.vera.lane;
    if pending == 0 {
        lane.task = None;
        lane.peak = 0;
        return;
    }
    lane.peak = lane.peak.max(pending);
    let task = lane.task.get_or_insert_with(|| s.activity.start("vera", "Embedding photos", Cancel::No));
    task.progress(lane.peak.saturating_sub(pending) as u64, lane.peak as u64);
}

/// Write new vectors to the cache file every few seconds, and when the work has run out.
fn save_if_due(s: &mut Session, pending: usize) {
    let lane = &mut s.vera.lane;
    if lane.store.has_unsaved() && (pending == 0 || lane.saved_at.is_none_or(|t| t.elapsed() >= SAVE_EVERY)) {
        if let Err(e) = lane.store.save() {
            log::warn!("vera: could not save the picture embeddings: {e}");
        }
        lane.saved_at = Some(Instant::now());
    }
}

/// Take in what the worker has finished.
#[cfg(not(target_arch = "wasm32"))]
fn collect(s: &mut Session) {
    use super::worker::Failure;
    let finished = s.vera.lane.engine.worker.as_ref().map(super::worker::Worker::finished).unwrap_or_default();
    for d in finished {
        let lane = &mut s.vera.lane;
        let photos = lane.in_flight.remove(&d.key).unwrap_or(1);
        if d.epoch != lane.engine.epoch {
            continue;
        }
        match d.result {
            Ok(v) => {
                if lane.store.insert(d.key.clone(), &v) {
                    lane.embedded_known = lane.embedded_known.saturating_add(photos);
                } else {
                    lane.skipped.insert(d.key);
                }
            }
            Err(Failure::Picture(e)) => {
                log::debug!("vera: photo {}: {e}", d.photo.0);
                lane.skipped.insert(d.key);
            }
            Err(Failure::Model(e)) => {
                // not the picture's fault: it is embedded once the model works again
                fail(s, e);
            }
        }
    }
}

/// The lane's part of one `vera.pump` call: `{state, pending, embedded, photos, inFlight}`.
pub(crate) fn pump(s: &mut Session, pace: Pace) -> Value {
    #[cfg(target_arch = "wasm32")]
    let _ = pace;
    #[cfg(not(target_arch = "wasm32"))]
    {
        collect(s);
        let st = state(s);
        let runnable = st == LaneState::Live || matches!(st, LaneState::Error(_)) && s.vera.lane.engine.retry_at.is_some_and(|t| Instant::now() >= t);
        let per_call = pace.per_call();
        if runnable && per_call > 0 {
            feed(s, pace, per_call);
        }
        if !runnable && s.vera.lane.in_flight.is_empty() {
            s.vera.lane.pending_known = 0;
        }
    }
    let lane = &s.vera.lane;
    let pending = lane.pending_known.saturating_add(lane.in_flight.values().sum::<usize>());
    track(s, pending);
    save_if_due(s, pending);
    let lane = &s.vera.lane;
    json!({
        "state": state(s).name(),
        "pending": pending,
        "embedded": lane.embedded_known,
        "photos": lane.photos_known,
        "inFlight": lane.in_flight.len(),
    })
}

/// Hand the worker up to `per_call` more pictures (at most [`MAX_IN_FLIGHT`] out at once).
#[cfg(not(target_arch = "wasm32"))]
fn feed(s: &mut Session, pace: Pace, per_call: usize) {
    let lane = &s.vera.lane;
    let stale = lane.queue_stamp != Some(s.catalog.revision);
    if lane.queue.is_empty() && lane.in_flight.is_empty() && stale && lane.queued_at.is_none_or(|t| t.elapsed() >= REQUEUE_EVERY) {
        refresh_queue(s);
    }
    if s.vera.lane.queue.is_empty() {
        return;
    }
    let Ok(m) = model(s) else { return };
    if s.vera.lane.engine.worker.is_none() {
        s.vera.lane.engine.worker = super::worker::Worker::start(m);
        if s.vera.lane.engine.worker.is_none() {
            fail(s, "could not start the embedding thread".into());
            return;
        }
    }
    let scan_pace = match pace {
        Pace::Pause => crate::faces_worker::Pace::Pause,
        Pace::Light => crate::faces_worker::Pace::Light,
        Pace::Normal => crate::faces_worker::Pace::Normal,
        Pace::Full => crate::faces_worker::Pace::Full,
    };
    let mut fed = 0;
    while fed < per_call && s.vera.lane.in_flight.len() < MAX_IN_FLIGHT {
        let Some(next) = s.vera.lane.queue.pop() else { break };
        let lane = &mut s.vera.lane;
        lane.pending_known = lane.pending_known.saturating_sub(next.ids.len());
        if lane.store.contains(&next.key) {
            // embedded since the look (`photo.similar`)
            lane.embedded_known = lane.embedded_known.saturating_add(next.ids.len());
            continue;
        }
        if lane.skipped.contains(&next.key) || lane.in_flight.contains_key(&next.key) {
            continue;
        }
        fed += 1;
        let prepared = next.ids.iter().find_map(|id| super::worker::prepare(s, *id).map(|p| (*id, p)));
        let lane = &mut s.vera.lane;
        let Some((photo, (render, preview))) = prepared else {
            lane.skipped.insert(next.key);
            continue;
        };
        let job = super::worker::Job { epoch: lane.engine.epoch, key: next.key.clone(), photo, render, preview, pace: scan_pace };
        if lane.engine.worker.as_ref().is_some_and(|w| w.submit(job)) {
            lane.in_flight.insert(next.key, next.ids.len());
        } else {
            // the thread is gone: start another on the next call
            lane.engine.worker = None;
            lane.queue.push(next);
            break;
        }
    }
}

/// Stop a running model download. False when none runs.
pub(crate) fn cancel_download(s: &Session) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    return s.vera.lane.engine.download.cancel();
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        false
    }
}

/// The model download's progress, while it runs: `{done, total}`.
pub(crate) fn download_json(s: &Session) -> Option<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let d = s.vera.lane.engine.download.status();
        if d.running {
            return Some(json!({"done": d.done, "total": d.total, "file": d.file}));
        }
    }
    let _ = s;
    None
}

/// The last download's error, if it failed.
pub(crate) fn download_error(s: &Session) -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    return s.vera.lane.engine.download.status().error;
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        None
    }
}

/// The user's own mirror list for the model (beside the model folder).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn mirrors_file(s: &Session) -> Option<PathBuf> {
    s.vera_model_dir.as_ref().and_then(|d| d.parent()).map(|p| p.join(MIRRORS_FILE))
}
