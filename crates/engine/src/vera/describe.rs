//! The Describe lane in the session (`docs/vera.md` → Phase 3): words proposed for photos, reviewed before anything is
//! kept ([`super::review`]).
//!
//! - **Where from.** An OpenAI-compatible endpoint the user configures (`vera.describe.config`, [`super::remote`]:
//!   loopback over `http`, any other host over `https` with the user's consent), and the zero-shot keywords of the
//!   shipped SigLIP 2 model ([`super::keywords`]): a remote proposal's keywords are ours first, then the endpoint's;
//!   `photo.describe {keywordsOnly: true}` needs no endpoint at all.
//! - **The queue.** `photo.describe {ids?}` adds photos (the selection by default, at most 10,000); `vera.pump` hands
//!   them to the worker threads at most [`super::remote::THREADS`] at a time, whatever the pace (describing is an
//!   explicit request), except that Pause stops new requests from being sent. Results go to the review queue, never to
//!   a photo. The activity stack shows "Describing photos" (`kind "describe"`, the host as its detail) with a ✕ that
//!   stops the run: nothing new is sent, what is in flight is abandoned.
//! - **Preferences.** The endpoint, the model name and the consent are the library's (`prefs.json` → `vera.describe`);
//!   the token is not a preference and is never in `prefs.json` (see `remote.rs`).
//!
//! The web build runs none of it: the lane reports `disabled`.

#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_variables, unused_mut, unused_imports))]

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use super::review::{self, ContextualQueue, Proposal, Review};
use super::{Pace, eligible};
use crate::activity::{Cancel, TaskGuard};
use crate::cmd::{CommandSpec, always, bad, cmd};
use crate::media::content_key;
use crate::{Result, Session};

/// The endpoint the user configured, saved with the library (`prefs.json` → `vera.describe`). The token is not here.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct Prefs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The user agreed to send photos to a host that is not this computer.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub allow_remote_host: bool,
}

impl Prefs {
    pub fn is_default(&self) -> bool {
        *self == Prefs::default()
    }
}

/// A photo waiting to be described.
struct Queued {
    photo: PhotoId,
    #[cfg(not(target_arch = "wasm32"))]
    remote: Option<Arc<super::remote::Remote>>,
}

/// One run of `photo.describe` calls: its activity row and its cancel flag.
struct Run {
    cancel: Arc<AtomicBool>,
    task: TaskGuard,
    total: usize,
    /// What the row names: the endpoint's host, or the keyword model.
    host: String,
}

/// The Describe lane's part of [`super::State`].
#[derive(Default)]
pub(crate) struct Describe {
    pub prefs: Prefs,
    pub review: Review,
    queue: VecDeque<Queued>,
    in_flight: HashSet<PhotoId>,
    /// Photos that could not be described this session, with why.
    pub failed: BTreeMap<PhotoId, String>,
    run: Option<Run>,
    pub contextual: ContextualQueue,
    pub keywords: super::keywords::KeywordState,
    /// A failure that stops the lane (the endpoint refused the token).
    pub error: Option<String>,
    /// The worker threads (they outlive a library change; what they finish for another library is dropped).
    #[cfg(not(target_arch = "wasm32"))]
    pool: Option<super::remote::Pool>,
    epoch: u64,
}

impl Describe {
    /// Another library: its own queue and proposals; the worker threads and the keyword vectors stay.
    pub fn library_changed(&mut self) -> Describe {
        Describe {
            keywords: self.keywords.library_changed(),
            #[cfg(not(target_arch = "wasm32"))]
            pool: self.pool.take(),
            epoch: self.epoch.wrapping_add(1),
            ..Describe::default()
        }
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

/// Where the Describe lane stands.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DescribeState {
    Live {
        host: String,
    },
    /// No endpoint (or no model name): `true` when the endpoint is set and only the model name is missing.
    NeedsEndpoint(bool),
    NeedsToken {
        host: String,
    },
    /// The web build: the remote path needs the native HTTP client, and the keyword proposals the model.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    NeedsDesktop,
    Error(String),
}

impl DescribeState {
    pub fn name(&self) -> &'static str {
        match self {
            DescribeState::Live { .. } => "live",
            DescribeState::NeedsEndpoint(_) => "needs_endpoint",
            DescribeState::NeedsToken { .. } => "needs_token",
            DescribeState::NeedsDesktop => "needs_desktop",
            DescribeState::Error(_) => "error",
        }
    }

    pub fn host(&self) -> Option<&str> {
        match self {
            DescribeState::Live { host } | DescribeState::NeedsToken { host } => Some(host),
            _ => None,
        }
    }

    pub fn reason(&self) -> String {
        match self {
            DescribeState::Live { host } => format!("Describing photos with the endpoint at {host}; every proposal waits for review (describe.list)."),
            DescribeState::NeedsEndpoint(false) => "Describing photos in words needs an OpenAI-compatible endpoint (llama.cpp, Ollama, vLLM or LM Studio on this computer): set one with vera.describe.config {endpoint, model}. Keyword proposals from the search model work without one (photo.describe {keywordsOnly: true}).".into(),
            DescribeState::NeedsEndpoint(true) => "Describing photos needs the endpoint's model name too: vera.describe.config {model}.".into(),
            DescribeState::NeedsToken { host } => format!("The endpoint at {host} is another computer and needs an access token: vera.describe.config {{token}}."),
            DescribeState::NeedsDesktop => "Describing photos needs the desktop app for now: the endpoint is reached through the desktop's HTTP client, which the browser build does not have, and keyword proposals need the search model the browser cannot run".into(),
            DescribeState::Error(e) => format!("Describing photos cannot run: {e}"),
        }
    }
}

/// The endpoint, the model and the token to describe with, checked; or why there are none.
#[cfg(not(target_arch = "wasm32"))]
fn remote(s: &Session) -> std::result::Result<super::remote::Remote, DescribeState> {
    use super::remote;
    let prefs = &s.vera.describe.prefs;
    let Some(ep) = prefs.endpoint.as_deref() else { return Err(DescribeState::NeedsEndpoint(false)) };
    let url = remote::endpoint(ep, prefs.allow_remote_host).map_err(DescribeState::Error)?;
    let Some(model) = prefs.model.as_deref() else { return Err(DescribeState::NeedsEndpoint(true)) };
    let path = remote::token_path(s.vera_model_dir.as_deref());
    let token = remote::read_token(path.as_deref(), remote::token_env()).map_err(DescribeState::Error)?;
    if token.is_none() && !url.is_loopback() {
        return Err(DescribeState::NeedsToken { host: remote::host_of(&url) });
    }
    if let Some(e) = &s.vera.describe.error {
        return Err(DescribeState::Error(e.clone()));
    }
    Ok(remote::Remote::new(&url, model, token))
}

pub(crate) fn state(s: &Session) -> DescribeState {
    #[cfg(target_arch = "wasm32")]
    return DescribeState::NeedsDesktop;
    #[cfg(not(target_arch = "wasm32"))]
    match remote(s) {
        Ok(r) => DescribeState::Live { host: r.host.clone() },
        Err(st) => st,
    }
}

/// `vera.status` → `lanes.describe`.
pub(crate) fn status_json(s: &Session) -> Value {
    let st = state(s);
    let d = &s.vera.describe;
    let mut out = json!({
        "state": st.name(),
        "reason": st.reason(),
        "pending": d.queue.len() + d.in_flight.len(),
        "proposals": d.review.len(),
    });
    if let Some(h) = st.host() {
        out["host"] = json!(h);
    }
    out
}

/// What `vera.describe.config` reports (never the token, only whether one is set).
fn config_json(s: &Session) -> Value {
    let p = &s.vera.describe.prefs;
    #[cfg(not(target_arch = "wasm32"))]
    let token_set = {
        let path = super::remote::token_path(s.vera_model_dir.as_deref());
        super::remote::read_token(path.as_deref(), super::remote::token_env()).is_ok_and(|t| t.is_some())
    };
    #[cfg(target_arch = "wasm32")]
    let token_set = false;
    let st = state(s);
    json!({
        "endpoint": p.endpoint,
        "model": p.model,
        "allowRemoteHost": p.allow_remote_host,
        "tokenSet": token_set,
        "state": st.name(),
        "reason": st.reason(),
    })
}

/// An optional text parameter that may be cleared: absent is `None`, null is `Some(None)`.
fn text_or_null(p: &Value, key: &str, max: usize, c: &str) -> Result<Option<Option<String>>> {
    match p.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(t)) => {
            let t = t.trim();
            if t.chars().count() > max || t.chars().any(char::is_control) {
                return Err(bad(c, format!("`{key}` is at most {max} characters, without line breaks")));
            }
            Ok(Some((!t.is_empty()).then(|| t.to_string())))
        }
        Some(_) => Err(bad(c, format!("`{key}` must be text or null"))),
    }
}

/// `vera.describe.config {endpoint?, model?, allowRemoteHost?, token?}`: where Describe sends photos. Not journaled
/// (it can carry the token).
fn config(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "vera.describe.config";
    #[cfg(target_arch = "wasm32")]
    {
        return Err(bad(C, DescribeState::NeedsDesktop.reason()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use super::remote;
        let endpoint = text_or_null(p, "endpoint", remote::MAX_ENDPOINT, C)?;
        let model = text_or_null(p, "model", remote::MAX_MODEL_NAME, C)?;
        let allow = match p.get("allowRemoteHost") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_bool().ok_or_else(|| bad(C, "`allowRemoteHost` must be true or false"))?),
        };
        let token: Option<Option<String>> = match p.get("token") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(Value::String(t)) if t.trim().is_empty() => Some(None),
            Some(Value::String(t)) => {
                remote::valid_token(t.trim()).map_err(|e| bad(C, e))?;
                Some(Some(t.trim().to_string()))
            }
            Some(_) => return Err(bad(C, "`token` must be text or null")),
        };
        let mut next = s.vera.describe.prefs.clone();
        if let Some(e) = endpoint {
            next.endpoint = e;
        }
        if let Some(m) = model {
            next.model = m;
        }
        if let Some(a) = allow {
            next.allow_remote_host = a;
        }
        // the policy holds for what would be saved: http only to this computer, another host only with consent
        if let Some(ep) = &next.endpoint {
            remote::endpoint(ep, next.allow_remote_host).map_err(|e| bad(C, e))?;
        }
        if let Some(t) = token {
            let path = remote::token_path(s.vera_model_dir.as_deref())
                .ok_or_else(|| bad(C, "this session has nowhere to keep a token (set LIGHTCRAFT_DESCRIBE_TOKEN instead)"))?;
            remote::write_token(&path, t.as_deref()).map_err(|e| bad(C, e))?;
            s.vera.describe.error = None;
        }
        if next != s.vera.describe.prefs {
            let endpoint_moved = next.endpoint != s.vera.describe.prefs.endpoint || next.model != s.vera.describe.prefs.model;
            s.vera.describe.prefs = next;
            s.vera.describe.error = None;
            if endpoint_moved && let Some(pool) = &s.vera.describe.pool {
                pool.shared.reset_format();
            }
            s.save_prefs()?;
        }
        Ok(config_json(s))
    }
}

/// `photo.describe {ids?, keywordsOnly?}`: queue photos for Describe (the selection by default).
fn describe(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.describe";
    super::ensure_open(s);
    let ids = super::commands::targets_param(s, p, C)?;
    if ids.is_empty() {
        return Err(bad(C, "no photos"));
    }
    let keywords_only = match p.get("keywordsOnly") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(C, "`keywordsOnly` must be true or false"))?,
    };
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (ids, keywords_only);
        return Err(bad(C, DescribeState::NeedsDesktop.reason()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let remote = if keywords_only {
            // the keywords come from the vector lane's model, or its vectors cached beside the library
            if super::keywords::matrix(s).ok().flatten().is_none() && super::lane::state(s) != super::lane::LaneState::Live {
                return Err(bad(C, format!("keyword proposals need the SigLIP 2 model: {}", super::lane::state(s).reason())));
            }
            None
        } else {
            match remote(s) {
                Ok(r) => Some(Arc::new(r)),
                Err(st) => return Err(bad(C, st.reason())),
            }
        };
        let d = &s.vera.describe;
        if d.queue.len().saturating_add(d.in_flight.len()).saturating_add(d.review.len()).saturating_add(ids.len()) > review::MAX_PROPOSALS {
            return Err(bad(C, format!("at most {} photos can wait for describing and review at once; review some first", review::MAX_PROPOSALS)));
        }
        let waiting: HashSet<PhotoId> = d.queue.iter().map(|q| q.photo).chain(d.in_flight.iter().copied()).collect();
        let (mut queued, mut already, mut skipped) = (0usize, 0usize, Vec::new());
        let mut seen = HashSet::new();
        for id in ids {
            if !seen.insert(id) {
                continue;
            }
            match s.catalog.photo(id) {
                None => skipped.push(json!({"id": id.0, "error": "no such photo"})),
                Some(ph) if !eligible(ph) => skipped.push(json!({"id": id.0, "error": "not a still photo in the library"})),
                Some(_) if waiting.contains(&id) => already += 1,
                Some(_) => {
                    s.vera.describe.failed.remove(&id);
                    s.vera.describe.queue.push_back(Queued { photo: id, remote: remote.clone() });
                    queued += 1;
                }
            }
        }
        if queued > 0 {
            let host = remote.as_ref().map(|r| r.host.clone());
            let d = &mut s.vera.describe;
            match &mut d.run {
                Some(run) if !run.cancel.load(Ordering::Relaxed) => run.total = run.total.saturating_add(queued),
                _ => {
                    let cancel = Arc::new(AtomicBool::new(false));
                    let task = s.activity.start("describe", "Describing photos", Cancel::Flag(cancel.clone()));
                    let host = host.unwrap_or_else(|| "SigLIP 2 keywords".into());
                    task.detail(&host);
                    d.run = Some(Run { cancel, task, total: queued.saturating_add(d.in_flight.len()), host });
                }
            }
            track(s);
        }
        let mut out = json!({"queued": queued, "already": already, "skipped": skipped, "keywordsOnly": keywords_only});
        if let Some(r) = &remote {
            out["host"] = json!(r.host);
            out["model"] = json!(r.model);
        }
        Ok(out)
    }
}

/// The run's activity row: done of total.
fn track(s: &mut Session) {
    let d = &mut s.vera.describe;
    let left = d.queue.len().saturating_add(d.in_flight.len());
    if let Some(run) = &d.run {
        run.task.progress(run.total.saturating_sub(left) as u64, run.total as u64);
    }
    if left == 0 {
        d.run = None;
    }
}

/// The proposal a finished job makes: the endpoint's fields with our keywords first, or our keywords alone.
#[cfg(not(target_arch = "wasm32"))]
fn proposal(photo: PhotoId, o: super::remote::Outcome) -> Option<Proposal> {
    let ours: Vec<String> = o.local.as_ref().map(|l| l.keywords.iter().map(|k| k.keyword.clone()).collect()).unwrap_or_default();
    let mut keywords: Vec<String> = Vec::new();
    let theirs = o.described.as_ref().map(|d| d.keywords.clone()).unwrap_or_default();
    for k in ours.into_iter().chain(theirs) {
        if !keywords.iter().any(|x| lightcraft_catalog::keywords::same(x, &k)) {
            keywords.push(k);
        }
    }
    let at = lightcraft_catalog::rules::now();
    let some = |t: &str| (!t.is_empty()).then(|| t.to_string());
    let (fields, source) = match &o.described {
        Some(d) => (
            review::Fields {
                title: some(&d.title),
                caption: some(&d.caption),
                alt_text: some(&d.alt_text),
                keywords: (!keywords.is_empty()).then_some(keywords),
            },
            review::Source { kind: "remote".into(), model: o.model.clone().unwrap_or_default(), host: o.host.clone() },
        ),
        None => (
            review::Fields { keywords: (!keywords.is_empty()).then_some(keywords), ..Default::default() },
            review::Source { kind: "local".into(), model: super::lane::MODEL_ID.into(), host: None },
        ),
    };
    if fields.names().is_empty() {
        return None;
    }
    Some(Proposal { photo: photo.0, fields, source, at })
}

/// Take in what the worker threads finished.
#[cfg(not(target_arch = "wasm32"))]
fn collect(s: &mut Session) {
    let finished = s.vera.describe.pool.as_ref().map(super::remote::Pool::finished).unwrap_or_default();
    for done in finished {
        let d = &mut s.vera.describe;
        if done.epoch != d.epoch {
            continue;
        }
        d.in_flight.remove(&done.photo);
        if let Some(v) = done.vector
            && !s.vera.lane.store.contains(&done.key)
        {
            s.vera.lane.store.insert(done.key.clone(), &v);
        }
        let d = &mut s.vera.describe;
        match done.result {
            Ok(o) => match proposal(done.photo, o) {
                Some(p) => {
                    if !d.review.insert(p) {
                        d.failed.insert(done.photo, "the review queue is full".into());
                    }
                }
                None => {
                    d.failed.insert(done.photo, "nothing to propose".into());
                }
            },
            Err(e) if e == super::remote::CANCELLED => {}
            Err(e) => {
                log::info!("vera: describe: photo {}: {e}", done.photo.0);
                if e.contains("HTTP 401") || e.contains("HTTP 403") {
                    // the token is wrong for every photo: the lane stops until it is set again
                    d.error = Some(e.clone());
                }
                d.failed.insert(done.photo, e);
            }
        }
    }
}

/// Hand the worker threads more photos, up to [`super::remote::THREADS`] in flight.
#[cfg(not(target_arch = "wasm32"))]
fn feed(s: &mut Session) {
    use super::remote::{IMAGE_EDGE, Job, KeywordJob, Pool, THREADS};
    if s.vera.describe.queue.is_empty() || s.vera.describe.in_flight.len() >= THREADS {
        return;
    }
    // the zero-shot keywords' vectors (built off this thread the first time: jobs wait for them)
    let lane_live = super::lane::state(s) == super::lane::LaneState::Live;
    let matrix = match super::keywords::matrix(s) {
        Ok(m) => m,
        Err(e) => {
            log::warn!("vera: the keyword vocabulary could not be embedded: {e}");
            None
        }
    };
    if matrix.is_none() && s.vera.describe.keywords.building() {
        // the first run embeds the vocabulary (a few minutes, once): the row says what it waits for
        if let Some(run) = &s.vera.describe.run {
            run.task.detail("SigLIP 2 vocabulary");
        }
        return;
    }
    if let Some(run) = &s.vera.describe.run {
        run.task.detail(&run.host);
    }
    let calibration = matrix.as_ref().and_then(|m| super::keywords::calibration(s, m));
    let model = if lane_live { super::lane::model(s).ok() } else { None };
    if s.vera.describe.pool.is_none() {
        s.vera.describe.pool = Pool::start();
        if s.vera.describe.pool.is_none() {
            log::warn!("vera: could not start the describe threads");
            return;
        }
    }
    let Some(cancel) = s.vera.describe.run.as_ref().map(|r| r.cancel.clone()) else { return };
    while s.vera.describe.in_flight.len() < THREADS {
        let Some(q) = s.vera.describe.queue.pop_front() else { break };
        let Some(photo) = s.catalog.photo(q.photo).filter(|p| eligible(p)).cloned() else {
            s.vera.describe.failed.insert(q.photo, "no such photo".into());
            continue;
        };
        let key = content_key(&photo);
        let vector = s.vera.lane.store.get(&key).map(<[f32]>::to_vec);
        let render = match &q.remote {
            Some(_) => match s.preview_job(q.photo, IMAGE_EDGE, IMAGE_EDGE, false, &lightcraft_develop::DevelopSettings::default()) {
                Some(r) => Some(r),
                None => {
                    s.vera.describe.failed.insert(q.photo, "the photo cannot be rendered".into());
                    continue;
                }
            },
            None => None,
        };
        let keywords = matrix.as_ref().and_then(|m| {
            // without a vector, the worker embeds the picture: it needs the model for that
            (vector.is_some() || model.is_some()).then(|| KeywordJob {
                vocab: s.vera.describe.keywords.vocab.clone(),
                matrix: m.clone(),
                calibration: calibration.clone(),
                vector: vector.clone(),
                model: if vector.is_none() { model.clone() } else { None },
                limit: super::keywords::MAX_KEYWORDS,
            })
        });
        if q.remote.is_none() && keywords.is_none() {
            s.vera.describe.failed.insert(q.photo, "keyword proposals need the SigLIP 2 model and the photo's embedding".into());
            continue;
        }
        let embed_render = if q.remote.is_none() && vector.is_none() { super::worker::prepare(s, q.photo) } else { None };
        let job = Job {
            epoch: s.vera.describe.epoch,
            photo: q.photo,
            key,
            render,
            embed_render,
            remote: q.remote.clone(),
            keywords,
            cancel: cancel.clone(),
        };
        let d = &mut s.vera.describe;
        if d.pool.as_ref().is_some_and(|p| p.submit(job)) {
            d.in_flight.insert(q.photo);
        } else {
            // the threads are gone: start them again on the next call
            d.pool = None;
            d.queue.push_front(q);
            break;
        }
    }
}

/// The Describe lane's part of one `vera.pump` call: `{pending, inFlight, proposals}`.
pub(crate) fn pump(s: &mut Session, pace: Pace) -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    {
        collect(s);
        if let Some(pool) = &s.vera.describe.pool {
            pool.shared.paused.store(pace == Pace::Pause, Ordering::Relaxed);
        }
        let cancelled = s.vera.describe.run.as_ref().is_some_and(|r| r.cancel.load(Ordering::Relaxed));
        if cancelled {
            // nothing new is sent; what is in flight comes back cancelled and is dropped
            s.vera.describe.queue.clear();
            s.vera.describe.in_flight.clear();
            s.vera.describe.epoch = s.vera.describe.epoch.wrapping_add(1);
        } else if pace != Pace::Pause {
            feed(s);
        }
        if pace != Pace::Pause {
            review::pump_contextual(s);
        }
    }
    track(s);
    let busy = s.vera.describe.run.is_some();
    s.vera.describe.review.save_if_due(!busy);
    let d = &s.vera.describe;
    json!({"pending": d.queue.len(), "inFlight": d.in_flight.len(), "proposals": d.review.len()})
}

/// `photo.keywords`, here so the lane's commands are listed together.
fn keywords(s: &mut Session, p: &Value) -> Result<Value> {
    super::keywords::photo_keywords(s, p)
}

/// Open the review queue of the library whose files are in `files` (dropping proposals of photos no longer in the
/// catalog).
pub(crate) fn open(s: &mut Session, files: &super::sidecar::Sidecars) {
    if !s.vera.describe.review.files().same_place(files) {
        let live: HashSet<u64> = s.catalog.photos().map(|p| p.id.0).collect();
        s.vera.describe.review = Review::load(files.clone(), &live);
    }
}

/// Map of the queue for tests.
#[cfg(test)]
pub(crate) fn queued(s: &Session) -> Vec<PhotoId> {
    s.vera.describe.queue.iter().map(|q| q.photo).collect()
}

pub fn specs() -> Vec<CommandSpec> {
    let mut v = vec![
        cmd!(
            query "vera.describe.config",
            "Describe Settings",
            [],
            None,
            "{endpoint?: url | null, model?: text | null, allowRemoteHost?: bool, token?: text | null} → {endpoint, model, allowRemoteHost, tokenSet, state, reason} — where Describe sends photos: an OpenAI-compatible endpoint (POST …/v1/chat/completions; llama.cpp, Ollama, vLLM, LM Studio). Plain http:// only to this computer; any other host needs https:// and allowRemoteHost: true, given only once the user agreed that the photos' pixels leave this computer. The token is kept in a private file (mode 0600) beside the model folders and is never shown: only tokenSet. Saved with the library's preferences; not journaled",
            always,
            config
        ),
        cmd!(
            query "photo.describe",
            "Describe Photos",
            [],
            None,
            "{ids?: [photo ids] (the selection, at most 10000), keywordsOnly?: bool (false)} → {queued, already, skipped: [{id, error}], keywordsOnly, host?, model?} — propose a title, caption, alt text and keywords for these photos in the background (the endpoint of vera.describe.config, with the search model's own zero-shot keywords first), or keywords alone from the search model with keywordsOnly (no endpoint needed). Proposals wait in the review queue (describe.list) and nothing is written until describe.accept. Shown in the activity stack (its ✕ stops the run); vera.pump drives it, and pace pause holds back new requests",
            always,
            describe
        ),
        cmd!(
            query "photo.keywords",
            "Suggest Keywords",
            [],
            None,
            "{id?: photo (the active one), limit?: 1..15 (15)} → {id, keywords: [{keyword, facet, cosine, z?}], ceiling, calibrated} — the keywords the search model (SigLIP 2) proposes for one photo from its built-in vocabulary, zero-shot: scored against the photo's embedding, cut at the photo's own noise ceiling and the vocabulary's facet rules, per-keyword calibrated (z) once the library has 200 embeddings. Read-only: nothing is written. The first call embeds the vocabulary (about a minute, once)",
            always,
            keywords
        ),
    ];
    v.extend(review::specs());
    v
}
