//! The Describe lane's review queue (`docs/vera.md` → Phase 3): what Describe proposed, waiting for the user. Nothing
//! is written to a photo until `describe.accept`, which is one ordinary `photo.setMeta` commit ("Accept Description",
//! undoable) that also records where each accepted field came from ([`lightcraft_catalog::Provenance::Describe`]).
//!
//! The queue is kept beside the library as `vera-proposals.json` (`{"version": 1, "proposals": [...]}`, written
//! atomically, read up to [`MAX_FILE`]); one proposal per photo (a newer one replaces it); proposals of photos no longer
//! in the catalog are dropped when the library opens. An in-memory library keeps them for the session.
//!
//! Accepting also schedules the photo's *contextual* embedding (the text tower's vector of "title. caption. keywords")
//! and its *fused* one (the normalised sum of the picture's canonical vector and the contextual one), both kept in the
//! embedding sidecar by role; `photo.similar` prefers the fused vector when a photo has one.

// the web build makes no proposals (nothing describes there) and has no text tower
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use lightcraft_catalog::PhotoId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use web_time::Instant;

use super::sidecar::{Loaded, Sidecars};
use crate::cmd::{CommandSpec, always, bad, cmd};
use crate::media::content_key;
use crate::{Result, Session};

/// The queue's file in the library folder.
pub(crate) const FILE_NAME: &str = "vera-proposals.json";
/// Largest queue file read.
pub(crate) const MAX_FILE: usize = 16 << 20;
/// Most proposals kept (and so most photos one library can have waiting for review).
pub(crate) const MAX_PROPOSALS: usize = 10_000;
/// How often new proposals are written while a run brings them in (and whenever it ends).
const SAVE_EVERY: Duration = Duration::from_secs(3);
/// The fields a proposal can hold, as `describe.accept` names them.
pub(crate) const FIELDS: [&str; 4] = ["title", "caption", "alt_text", "keywords"];
/// Longest value an edit may give a text field, and most keywords.
const MAX_EDIT: usize = 2000;
const MAX_EDIT_KEYWORDS: usize = 200;
/// Contextual embeddings waiting for the text tower.
const MAX_CONTEXTUAL: usize = 10_000;

/// What a proposal says.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Fields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alt_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<String>>,
}

impl Fields {
    /// The names of the fields this proposal fills (non-empty).
    pub fn names(&self) -> Vec<&'static str> {
        let text = |v: &Option<String>| v.as_ref().is_some_and(|t| !t.is_empty());
        let mut out = Vec::new();
        if text(&self.title) {
            out.push("title");
        }
        if text(&self.caption) {
            out.push("caption");
        }
        if text(&self.alt_text) {
            out.push("alt_text");
        }
        if self.keywords.as_ref().is_some_and(|k| !k.is_empty()) {
            out.push("keywords");
        }
        out
    }
}

/// Who proposed it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Source {
    /// `remote` (an endpoint, with our keywords added) or `local` (keywords only, from the shipped model).
    pub kind: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
}

/// One photo's proposal.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Proposal {
    pub photo: u64,
    pub fields: Fields,
    pub source: Source,
    /// When it was proposed (ISO 8601).
    pub at: String,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct QueueFile {
    version: u32,
    proposals: Vec<Proposal>,
}

/// The proposals of the library in use.
#[derive(Default)]
pub(crate) struct Review {
    proposals: BTreeMap<u64, Proposal>,
    files: Sidecars,
    dirty: bool,
    saved_at: Option<Instant>,
}

impl Review {
    /// The proposals in the file kept in `files` whose photo is in `live` (none when it is missing, too large or not
    /// ours).
    pub fn load(files: Sidecars, live: &std::collections::HashSet<u64>) -> Review {
        let mut proposals = BTreeMap::new();
        let mut dropped = false;
        if let Loaded::Bytes(bytes) = files.read(FILE_NAME, MAX_FILE as u64) {
            match serde_json::from_slice::<QueueFile>(&bytes) {
                Ok(f) => {
                    for pr in f.proposals {
                        if live.contains(&pr.photo) && proposals.len() < MAX_PROPOSALS {
                            proposals.insert(pr.photo, pr);
                        } else {
                            dropped = true;
                        }
                    }
                }
                Err(e) => log::warn!("vera: {} is unreadable ({e}); starting with no proposals", files.describe(FILE_NAME)),
            }
        }
        Review { proposals, files, dirty: dropped, saved_at: None }
    }

    /// Where the file is kept.
    pub fn files(&self) -> &Sidecars {
        &self.files
    }

    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    pub fn get(&self, photo: u64) -> Option<&Proposal> {
        self.proposals.get(&photo)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Proposal> {
        self.proposals.values()
    }

    /// Keep a proposal (replacing the photo's earlier one); `false` when the queue is full.
    pub fn insert(&mut self, p: Proposal) -> bool {
        if self.proposals.len() >= MAX_PROPOSALS && !self.proposals.contains_key(&p.photo) {
            return false;
        }
        self.proposals.insert(p.photo, p);
        self.dirty = true;
        true
    }

    pub fn remove(&mut self, photo: u64) -> Option<Proposal> {
        let p = self.proposals.remove(&photo);
        self.dirty |= p.is_some();
        p
    }

    pub fn clear(&mut self) -> usize {
        let n = self.proposals.len();
        self.proposals.clear();
        self.dirty |= n > 0;
        n
    }

    /// Write the queue (atomically) when it changed; without a library it is kept for the session.
    pub fn save(&mut self) -> std::result::Result<(), String> {
        if !self.dirty {
            return Ok(());
        }
        if !self.files.persists() {
            self.dirty = false;
            return Ok(());
        }
        let file = QueueFile { version: 1, proposals: self.proposals.values().cloned().collect() };
        let bytes = serde_json::to_vec(&file).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_FILE {
            return Err(format!("the review queue is larger than {} MiB", MAX_FILE >> 20));
        }
        self.files.write_atomic(FILE_NAME, &bytes)?;
        self.dirty = false;
        self.saved_at = Some(Instant::now());
        Ok(())
    }

    /// [`Review::save`], at most every few seconds unless `now`.
    pub fn save_if_due(&mut self, now: bool) {
        if self.dirty
            && (now || self.saved_at.is_none_or(|t| t.elapsed() >= SAVE_EVERY))
            && let Err(e) = self.save()
        {
            log::warn!("vera: could not save the describe proposals: {e}");
        }
    }
}

impl Drop for Review {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            log::warn!("vera: could not save the describe proposals: {e}");
        }
    }
}

/// A photo's current values of `fields`, for the reviewer to compare with.
fn current(s: &Session, photo: u64, fields: &[&str]) -> Value {
    let Some(p) = s.catalog.photo(PhotoId(photo)) else { return json!({}) };
    let m = &p.meta;
    let mut out = serde_json::Map::new();
    for f in fields {
        let v = match *f {
            "title" => json!(m.title),
            "caption" => json!(m.caption),
            "alt_text" => json!(m.alt_text),
            "keywords" => json!(m.keywords),
            _ => continue,
        };
        out.insert((*f).to_string(), v);
    }
    Value::Object(out)
}

/// One proposal as the commands show it.
fn proposal_json(s: &Session, p: &Proposal) -> Value {
    let mut v = serde_json::to_value(p).unwrap_or_else(|_| json!({"photo": p.photo}));
    v["current"] = current(s, p.photo, &p.fields.names());
    v
}

/// `describe.list {limit?}`: the review queue, photos in id order, and the run's progress and failures.
fn list(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "describe.list";
    let limit = match p.get("limit") {
        None | Some(Value::Null) => MAX_PROPOSALS,
        Some(v) => {
            usize::try_from(v.as_u64().ok_or_else(|| bad(C, "`limit` must be a whole number"))?).unwrap_or(MAX_PROPOSALS).clamp(1, MAX_PROPOSALS)
        }
    };
    super::ensure_open(s);
    let rows: Vec<Value> =
        s.vera.describe.review.iter().filter(|pr| s.catalog.photo(PhotoId(pr.photo)).is_some()).take(limit).map(|pr| proposal_json(s, pr)).collect();
    let failed: Vec<Value> = s.vera.describe.failed.iter().map(|(id, e)| json!({"photo": id.0, "error": e})).collect();
    let d = &s.vera.describe;
    Ok(json!({
        "proposals": rows,
        "total": d.review.len(),
        "pending": d.pending(),
        "inFlight": d.in_flight(),
        "failed": failed,
    }))
}

/// A text field of an edit.
fn edit_text(c: &str, field: &str, v: &Value) -> Result<String> {
    let t = v.as_str().ok_or_else(|| bad(c, format!("edits.{field} must be text")))?;
    if t.chars().count() > MAX_EDIT {
        return Err(bad(c, format!("edits.{field} is longer than {MAX_EDIT} characters")));
    }
    Ok(t.to_string())
}

/// `describe.accept {id, fields?, edits?, replaceKeywords?}`: write the proposal's fields (or some of them, edited or
/// not) to the photo as one `photo.setMeta` commit, with their provenance; the proposal leaves the queue.
fn accept(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "describe.accept";
    super::ensure_open(s);
    let id = p.get("id").and_then(Value::as_u64).ok_or_else(|| bad(C, "`id` must be the photo id of a proposal"))?;
    if s.catalog.photo(PhotoId(id)).is_none() {
        return Err(bad(C, format!("no photo {id}")));
    }
    let proposal = s.vera.describe.review.get(id).cloned().ok_or_else(|| bad(C, format!("photo {id} has no proposal to accept")))?;
    let edits = match p.get("edits") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return Err(bad(C, "`edits` is {field: value}")),
    };
    for k in edits.keys() {
        if !FIELDS.contains(&k.as_str()) {
            return Err(bad(C, format!("edits: unknown field `{k}` ({})", FIELDS.join(", "))));
        }
    }
    let fields: Vec<String> = match p.get("fields") {
        None | Some(Value::Null) => {
            let mut f: Vec<String> = proposal.fields.names().into_iter().map(str::to_string).collect();
            for k in edits.keys() {
                if !f.contains(k) {
                    f.push(k.clone());
                }
            }
            f
        }
        Some(Value::Array(a)) => {
            let mut f = Vec::new();
            for v in a {
                let name =
                    v.as_str().filter(|n| FIELDS.contains(n)).ok_or_else(|| bad(C, format!("`fields` names some of {}", FIELDS.join(", "))))?;
                if !f.iter().any(|x| x == name) {
                    f.push(name.to_string());
                }
            }
            f
        }
        Some(_) => return Err(bad(C, "`fields` must be a list of field names")),
    };
    if fields.is_empty() {
        return Err(bad(C, "nothing to accept: the proposal has no fields"));
    }
    let replace = match p.get("replaceKeywords") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(C, "`replaceKeywords` must be true or false"))?,
    };
    let at = lightcraft_catalog::rules::now();
    let origin = json!({"kind": "describe", "model": proposal.source.model, "at": at});
    let mut params = serde_json::Map::new();
    params.insert("ids".into(), json!([id]));
    let mut provenance = serde_json::Map::new();
    for f in &fields {
        let proposed = match f.as_str() {
            "title" => proposal.fields.title.clone().map(Value::String),
            "caption" => proposal.fields.caption.clone().map(Value::String),
            "alt_text" => proposal.fields.alt_text.clone().map(Value::String),
            _ => proposal.fields.keywords.clone().map(|k| json!(k)),
        };
        let value = match edits.get(f) {
            Some(e) if f == "keywords" => {
                let a = e.as_array().ok_or_else(|| bad(C, "edits.keywords must be a list of keywords"))?;
                if a.len() > MAX_EDIT_KEYWORDS {
                    return Err(bad(C, format!("edits.keywords: at most {MAX_EDIT_KEYWORDS}")));
                }
                let k: Vec<String> = a.iter().map(|k| edit_text(C, f, k)).collect::<Result<_>>()?;
                json!(k)
            }
            Some(e) => json!(edit_text(C, f, e)?),
            None => proposed.filter(|v| v.as_str().is_none_or(|t| !t.is_empty())).ok_or_else(|| bad(C, format!("the proposal has no {f}")))?,
        };
        let key = match f.as_str() {
            "title" => "title",
            "caption" => "caption",
            "alt_text" => "altText",
            _ if replace => "keywords",
            _ => "addKeywords",
        };
        params.insert(key.into(), value);
        provenance.insert(f.clone(), origin.clone());
    }
    params.insert("provenance".into(), Value::Object(provenance));
    let changed = crate::cmd::library::set_meta(s, &Value::Object(params), "Accept Description")?;
    s.vera.describe.review.remove(id);
    s.vera.describe.review.save_if_due(true);
    schedule_contextual(s, PhotoId(id));
    Ok(json!({"id": id, "accepted": fields, "changed": changed["changed"], "proposals": s.vera.describe.review.len()}))
}

/// `describe.reject {id}`: drop a proposal, writing nothing.
fn reject(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "describe.reject";
    super::ensure_open(s);
    let id = p.get("id").and_then(Value::as_u64).ok_or_else(|| bad(C, "`id` must be the photo id of a proposal"))?;
    let removed = s.vera.describe.review.remove(id).is_some();
    s.vera.describe.review.save_if_due(true);
    Ok(json!({"id": id, "removed": removed, "proposals": s.vera.describe.review.len()}))
}

/// `describe.clear {}`: drop every proposal.
fn clear(s: &mut Session, _: &Value) -> Result<Value> {
    super::ensure_open(s);
    let n = s.vera.describe.review.clear();
    s.vera.describe.review.save_if_due(true);
    Ok(json!({"removed": n, "proposals": 0}))
}

/// The text the contextual embedding is made of: "title. caption. keywords" (empty parts left out).
pub(crate) fn contextual_text(m: &lightcraft_catalog::Meta) -> String {
    let mut parts: Vec<String> = Vec::new();
    for t in [&m.title, &m.caption] {
        let t = t.trim().trim_end_matches('.');
        if !t.is_empty() {
            parts.push(t.to_string());
        }
    }
    let keywords: Vec<String> = m.keywords.iter().map(|k| k.replace('|', " ")).collect();
    if !keywords.is_empty() {
        parts.push(keywords.join(", "));
    }
    let mut text = parts.join(". ");
    if !text.is_empty() {
        text.push('.');
    }
    text
}

/// Queue photo `id`'s contextual and fused embeddings (made by the pump once the text tower is at hand).
pub(crate) fn schedule_contextual(s: &mut Session, id: PhotoId) {
    let Some(p) = s.catalog.photo(id) else { return };
    let text = contextual_text(&p.meta);
    if text.is_empty() {
        return;
    }
    let key = content_key(p);
    let q = &mut s.vera.describe.contextual;
    q.retain(|(other, _, _)| *other != id);
    if q.len() >= MAX_CONTEXTUAL {
        q.pop_front();
    }
    q.push_back((id, key, text));
}

/// One waiting contextual embedding, when the text tower can make it: `true` when one was made.
pub(crate) fn pump_contextual(s: &mut Session) -> bool {
    let Some((id, key, text)) = s.vera.describe.contextual.front().cloned() else { return false };
    let v = match super::lane::query_vector(s, &text) {
        Ok(Some(v)) => v,
        // the lane is not running: kept for later
        Ok(None) => return false,
        Err(e) => {
            log::debug!("vera: photo {}: no contextual embedding: {e}", id.0);
            s.vera.describe.contextual.pop_front();
            return false;
        }
    };
    s.vera.describe.contextual.pop_front();
    let store = &mut s.vera.lane.store;
    if !store.insert_role(key.clone(), super::embeddings::ROLE_CONTEXTUAL, &v) {
        return false;
    }
    if let (Some(c), Some(t)) = (store.get(&key), store.get_role(&key, super::embeddings::ROLE_CONTEXTUAL)) {
        let sum: Vec<f32> = c.iter().zip(t).map(|(a, b)| a + b).collect();
        store.insert_role(key, super::embeddings::ROLE_FUSED, &sum);
    }
    true
}

/// Waiting contextual embeddings.
pub(crate) type ContextualQueue = VecDeque<(PhotoId, String, String)>;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "describe.list",
            "Review Descriptions",
            [],
            None,
            "{limit?: 1..10000} → {proposals: [{photo, fields: {title?, caption?, alt_text?, keywords?: [..]}, source: {kind: remote | local, model, host?}, at, current: {the photo's values of those fields}}], total, pending, inFlight, failed: [{photo, error}]} — the descriptions Describe proposed, waiting for review (nothing is written to a photo until describe.accept), the photos still being described and those that failed this session",
            always,
            list
        ),
        cmd!(
            "describe.accept",
            "Accept Description",
            [],
            None,
            "{id: photo id, fields?: [title | caption | alt_text | keywords] (all the proposal has), edits?: {field: value}, replaceKeywords?: bool (false)} → {id, accepted, changed, proposals} — write a proposal's fields to its photo (edited where `edits` says) as one undoable Edit Info step, recording each field's provenance (describe, the model, the time); keywords are added to the photo's own unless replaceKeywords. The proposal leaves the queue",
            always,
            accept
        ),
        cmd!(
            query "describe.reject",
            "Reject Description",
            [],
            None,
            "{id: photo id} → {id, removed, proposals} — drop a proposal without writing anything",
            always,
            reject
        ),
        cmd!(
            query "describe.clear",
            "Clear Descriptions",
            [],
            None,
            "{} → {removed, proposals} — drop every proposal in the review queue",
            always,
            clear
        ),
    ]
}
