//! VERA's commands: `vera.status`, `vera.pump`, `vera.digest`, `search.text`, `photo.duplicates`, `photo.similar`,
//! `photo.ignorePair` / `photo.unignorePair` and `photo.stackDuplicate` (`docs/vera.md` → Commands); the model's are in
//! [`super::model`].
//!
//! Every parameter is hostile input: types are checked, lists and limits are capped at [`MAX_LIMIT`].

use std::collections::HashMap;

use lightcraft_catalog::{PhotoId, Sort, SortKey};
use lightcraft_vera::{Sameness, thresholds};
use serde_json::{Value, json};

use super::lane::{self, LaneState};
use super::{Pace, digest_now, eligible, ensure_index, ensure_open};
use crate::cmd::{CommandSpec, always, bad, cmd};
use crate::media::content_key;
use crate::{Result, Session};

/// The most any list parameter or limit may ask for.
pub(crate) const MAX_LIMIT: usize = 10_000;
/// Longest search query.
const MAX_QUERY: usize = 2000;

/// Why the describe lane is not running, said the same way everywhere.
const DESCRIBE_REASON: &str = "Describing photos in words needs a captioning model; it is not part of this build (Phase 3).";

/// An optional count: absent or null is `default`, a number is clamped to 1..=[`MAX_LIMIT`], anything else is an error.
fn limit_param(p: &Value, key: &str, default: usize, c: &str) -> Result<usize> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad(c, format!("`{key}` must be a whole number from 1 to {MAX_LIMIT}")))?;
            Ok(usize::try_from(n).unwrap_or(MAX_LIMIT).clamp(1, MAX_LIMIT))
        }
    }
}

/// An optional flag: absent or null is `default`, anything but a boolean is an error.
fn bool_param(p: &Value, key: &str, default: bool, c: &str) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_bool().ok_or_else(|| bad(c, format!("`{key}` must be true or false"))),
    }
}

/// A photo id parameter: absent or null is `None`; a photo that is not in the catalog is an error.
fn photo_param(s: &Session, p: &Value, key: &str, c: &str) -> Result<Option<PhotoId>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let id = v.as_u64().map(PhotoId).ok_or_else(|| bad(c, format!("`{key}` must be a photo id")))?;
            if s.catalog.photo(id).is_none() {
                return Err(bad(c, format!("no photo {}", id.0)));
            }
            Ok(Some(id))
        }
    }
}

/// `ids` (at most [`MAX_LIMIT`] photo ids), else the selection.
fn targets_param(s: &Session, p: &Value, c: &str) -> Result<Vec<PhotoId>> {
    match p.get("ids") {
        None | Some(Value::Null) => Ok(s.targets(&json!({}))),
        Some(Value::Array(a)) => {
            if a.len() > MAX_LIMIT {
                return Err(bad(c, format!("at most {MAX_LIMIT} photos at a time")));
            }
            a.iter().map(|v| v.as_u64().map(PhotoId).ok_or_else(|| bad(c, "`ids` must be photo ids"))).collect()
        }
        Some(_) => Err(bad(c, "`ids` must be a list of photo ids")),
    }
}

/// One sidecar file for `vera.status`: its name, and where it is and how large when the library is on disk.
fn sidecar_json(name: &str, path: Option<&std::path::Path>, bytes: Option<u64>) -> Value {
    match path {
        Some(path) => json!({"name": name, "path": path.to_string_lossy(), "bytes": bytes.unwrap_or(0)}),
        None => json!({"name": name}),
    }
}

/// `vera.status {}`: what VERA knows about this library, and which lanes run in this build.
fn status(s: &mut Session, _: &Value) -> Result<Value> {
    ensure_open(s);
    let (mut photos, mut digested, mut skipped) = (0usize, 0usize, 0usize);
    for p in s.catalog.photos().filter(|p| eligible(p)) {
        photos += 1;
        let key = content_key(p);
        if s.vera.digests.contains(&key) {
            digested += 1;
        } else if s.vera.skipped.contains(&key) {
            skipped += 1;
        }
    }
    let sidecar = match s.vera.digests.path() {
        Some(path) => json!({"path": path.to_string_lossy(), "bytes": s.vera.digests.file_len().unwrap_or(0)}),
        None => json!({}),
    };
    let embeddings_name = super::embeddings::file_name(lane::MODEL_ID);
    let sidecars = json!([
        sidecar_json(super::digests::FILE_NAME, s.vera.digests.path(), s.vera.digests.file_len()),
        sidecar_json(&embeddings_name, s.vera.lane.store.path(), s.vera.lane.store.file_len()),
    ]);
    let (_, embedded, pending_embeddings) = lane::counts(s);
    let vector = lane::state(s);
    let loaded = s.vera.lane.engine.loaded();
    // the space's own background (random pairs of its pictures): a diagnostic, since queries are cut against their own
    // population; a mean that drifts towards 1 says the library's pictures all look alike to the model
    let space_background =
        s.vera.lane.store.background().map(|b| json!({"mean": round(b.mean, 4), "sigma": round(b.sigma, 4), "pictures": b.n_at_compute}));
    Ok(json!({
        "digested": digested,
        "photos": photos,
        "pendingDigests": photos.saturating_sub(digested).saturating_sub(skipped),
        "unreadable": skipped,
        "digests": s.vera.digests.len(),
        "embedded": embedded,
        "pendingEmbeddings": pending_embeddings,
        "unembeddable": lane::unreadable(s),
        "embeddings": s.vera.lane.store.len(),
        "spaceBackground": space_background,
        "model": {
            "id": lane::MODEL_ID,
            "installed": lane::installed(s),
            "loaded": loaded,
            // resident memory while loaded: the model once, plus one batch of inputs
            "memoryBytes": if loaded { lane::MODEL_BYTES + lane::BATCH_BYTES } else { 0 },
        },
        "indexDocs": s.vera.index.as_ref().map_or(0, |i| i.len()),
        "ignoredPairs": s.vera.ignored.len(),
        "lanes": {
            "keyword": "live",
            "digest": "live",
            "vector": {"state": vector.name(), "reason": vector.reason()},
            "describe": {"state": "disabled", "reason": DESCRIBE_REASON},
        },
        "sidecar": sidecar,
        "sidecars": sidecars,
    }))
}

/// `vera.pump {pace?}`: the per-frame background fingerprinting.
fn pump(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "vera.pump";
    let pace = match p.get("pace") {
        None | Some(Value::Null) => Pace::Normal,
        Some(v) => v.as_str().and_then(Pace::parse).ok_or_else(|| bad(C, "`pace` must be pause, light, normal or full"))?,
    };
    Ok(super::pump(s, pace))
}

/// `vera.digest {ids?}`: fingerprint these photos now (the selection by default).
fn digest(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "vera.digest";
    ensure_open(s);
    let ids = targets_param(s, p, C)?;
    let (mut digested, mut already, mut failed) = (0usize, 0usize, Vec::new());
    for id in ids {
        let Some(key) = s.catalog.photo(id).map(|ph| content_key(ph)) else {
            failed.push(json!({"id": id.0, "error": "no such photo"}));
            continue;
        };
        if s.vera.digests.contains(&key) {
            already += 1;
            continue;
        }
        match digest_now(s, id) {
            Ok(_) => digested += 1,
            Err(e) => failed.push(json!({"id": id.0, "error": e})),
        }
    }
    if let Err(e) = s.vera.digests.save() {
        log::warn!("vera: could not save the picture digests: {e}");
    }
    Ok(json!({"digested": digested, "already": already, "failed": failed}))
}

/// Round for a result row.
fn round(v: f32, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (f64::from(v) * f).round() / f
}

/// Which kind of query a noise ceiling is for. Both measure the ceiling on the query's own population (its cosines
/// with every picture): the space's random-pair background is on another scale than a text–image cosine (SigLIP 2:
/// about 0.0–0.15 against 0.5–0.9), and on a small or uniform library it sits so high that no picture ever clears it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Noise {
    /// An image query (`photo.similar`): the ceiling never rises above [`thresholds::EMBED_NEAR`], because a picture
    /// that near is a near duplicate by measurement and is never noise, however alike the rest of the library is.
    Image,
    /// A text query (`search.text`).
    Text,
}

/// The background of `query`'s own cosines with every vector of `space` (`None` under two vectors): the statistic of
/// [`lightcraft_vera::cutoff::background`] (median and MAD of the lower half), over this query's population.
pub(super) fn query_background(space: &lightcraft_vera::Space, query: &[f32]) -> Option<lightcraft_vera::cutoff::Background> {
    let c: Vec<f32> = space.nearest(query, space.len(), None).into_iter().map(|(_, c)| c).collect();
    lightcraft_vera::cutoff::background_from_scores(&c, space.len(), space.seed())
}

/// The vector lane's nearest pictures to `query` (a unit vector), as photos in rank order with their cosines, and the
/// noise ceiling applied (`None` when it did not engage). `exclude` leaves one picture out.
fn vector_hits(s: &mut Session, query: &[f32], limit: usize, exclude: Option<&str>, noise: Noise) -> (Vec<(PhotoId, f32)>, Option<f32>) {
    if s.vera.lane.store.is_empty() {
        return (Vec::new(), None);
    }
    let store = &s.vera.lane.store;
    let Some(space) = store.space() else { return (Vec::new(), None) };
    let bg = query_background(space, query);
    let exclude = exclude.and_then(|k| store.id_of(k));
    let (rows, ceiling) = match noise {
        Noise::Text => space.search(query, limit, exclude, bg.as_ref()),
        Noise::Image => {
            // the ceiling, capped: a near duplicate is never noise
            let candidates = space.len().saturating_sub(usize::from(exclude.is_some()));
            let cap = bg.as_ref().map(|b| lightcraft_vera::cutoff::ceiling(b, candidates).min(thresholds::EMBED_NEAR));
            let (mut rows, _) = space.search(query, limit, exclude, None);
            if let Some(c) = cap
                && candidates >= thresholds::MIN_SCORED_FOR_CUTOFF
            {
                rows.retain(|(_, s)| *s >= c);
            }
            (rows, cap.filter(|_| candidates >= thresholds::MIN_SCORED_FOR_CUTOFF))
        }
    };
    let by_key = lane::photos_by_key(s);
    let mut out = Vec::new();
    for (id, cos) in rows {
        let Some(ids) = store.key_of(id).and_then(|k| by_key.get(k)) else { continue };
        out.extend(ids.iter().map(|p| (*p, cos)));
    }
    out.truncate(limit);
    (out, ceiling)
}

/// `search.text {query, limit?, filter?}`: the keyword lane, fused with the vector lane when it runs.
fn search_text(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "search.text";
    let query = match p.get("query") {
        Some(Value::String(q)) => q.clone(),
        Some(_) => return Err(bad(C, "`query` must be text")),
        None => return Err(bad(C, "missing `query`")),
    };
    if query.chars().count() > MAX_QUERY {
        return Err(bad(C, format!("a query is at most {MAX_QUERY} characters")));
    }
    let limit = limit_param(p, "limit", 200, C)?;
    let filter = bool_param(p, "filter", true, C)?;
    ensure_index(s);
    ensure_open(s);
    let hits = s.vera.index.as_ref().map(|i| i.search(&query, limit)).unwrap_or_default();
    let (query_vec, warning) = if query.trim().is_empty() {
        (None, None)
    } else {
        match lane::query_vector(s, &query) {
            Ok(v) => (v, None),
            Err(e) => (None, Some(format!("Showing keyword matches only: {e}"))),
        }
    };
    let (photos, ids, cutoff, lanes): (Vec<Value>, Vec<PhotoId>, Option<f32>, Value) = match query_vec {
        None => {
            let photos = hits.iter().map(|h| json!({"id": h.doc, "score": round(h.score, 3), "lane": "keyword"})).collect();
            (photos, hits.iter().map(|h| PhotoId(h.doc)).collect(), None, json!(["keyword"]))
        }
        Some(qv) => {
            let (vector, ceiling) = vector_hits(s, &qv, limit, None, Noise::Text);
            let keyword_ids: Vec<u64> = hits.iter().map(|h| h.doc).collect();
            let vector_ids: Vec<u64> = vector.iter().map(|(id, _)| id.0).collect();
            let cosines: HashMap<u64, f32> = vector.iter().map(|(id, c)| (id.0, *c)).collect();
            let keyword_scores: HashMap<u64, f32> = hits.iter().map(|h| (h.doc, h.score)).collect();
            let mut fused = lightcraft_vera::rrf(&[&keyword_ids, &vector_ids], &[1.0, 1.0]);
            fused.truncate(limit);
            let photos = fused
                .iter()
                .map(|(id, score)| {
                    let (k, v) = (keyword_scores.get(id), cosines.get(id));
                    let lane = match (k, v) {
                        (Some(_), Some(_)) => "both",
                        (None, Some(_)) => "vector",
                        _ => "keyword",
                    };
                    let mut row = json!({"id": id, "score": round(*score, 6), "lane": lane});
                    if let Some(c) = v {
                        row["cosine"] = json!(round(*c, 4));
                    }
                    if let Some(k) = k {
                        row["keywordScore"] = json!(round(*k, 3));
                    }
                    row
                })
                .collect();
            (photos, fused.iter().map(|(id, _)| PhotoId(*id)).collect(), ceiling, json!(["keyword", "vector"]))
        }
    };
    // a search that found nothing leaves the view alone: an empty `only` would show everything
    let filtered = filter && !ids.is_empty();
    if filtered {
        s.filter.only = ids;
        s.sort = Sort { key: SortKey::Relevance, ascending: false, ..s.sort };
    }
    let mut out = json!({"photos": photos, "cutoff": cutoff.map(|c| round(c, 4)), "lanes": lanes, "filtered": filtered});
    if let Some(w) = warning {
        out["warning"] = json!(w);
    }
    Ok(out)
}

/// `photo.similar {id?, limit?, filter?}`: the pictures that look most like this one, by the vector lane.
fn similar(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.similar";
    let id = photo_param(s, p, "id", C)?.or_else(|| s.active()).ok_or_else(|| bad(C, "no photo"))?;
    let limit = limit_param(p, "limit", 50, C)?;
    let filter = bool_param(p, "filter", true, C)?;
    ensure_open(s);
    let key = s.catalog.photo(id).map(|ph| content_key(ph)).ok_or_else(|| bad(C, "no such photo"))?;
    let embedded_before = s.vera.lane.store.contains(&key);
    let mine = match lane::vector_of(s, id, true) {
        Ok(Some(v)) => v,
        Ok(None) => {
            let st = lane::state(s);
            let why = if st == LaneState::Live { "this photo could not be embedded".to_string() } else { st.reason() };
            return Err(bad(C, format!("the vector lane is not ready: {why}")));
        }
        Err(e) => return Err(bad(C, format!("the vector lane is not ready: {e}"))),
    };
    let (rows, cutoff) = vector_hits(s, &mine, limit, Some(&key), Noise::Image);
    let filtered = filter && !rows.is_empty();
    if filtered {
        s.filter.only = std::iter::once(id).chain(rows.iter().map(|(q, _)| *q)).collect();
        s.sort = Sort { key: SortKey::Relevance, ascending: false, ..s.sort };
    }
    let (_, _, pending) = lane::counts(s);
    let photos: Vec<Value> = rows.iter().map(|(q, c)| json!({"id": q.0, "cosine": round(*c, 4), "lane": "vector"})).collect();
    Ok(json!({
        "id": id.0,
        "photos": photos,
        "cutoff": cutoff.map(|c| round(c, 4)),
        "embedded": embedded_before,
        "pendingEmbeddings": pending,
        "filtered": filtered,
    }))
}

/// Order of the duplicate tiers in a list.
fn tier_rank(tier: &str) -> u8 {
    match tier {
        "bytes" => 0,
        "picture" => 1,
        "possible" => 2,
        _ => 3,
    }
}

/// One row of `photo.duplicates`.
struct DupRow {
    tier: &'static str,
    id: u64,
    distances: Option<(u32, u32)>,
    cosine: Option<f32>,
}

/// `photo.duplicates {id?, limit?, filter?}`: the photos that are the same picture as this one, by tier.
fn duplicates(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.duplicates";
    let id = photo_param(s, p, "id", C)?.or_else(|| s.active()).ok_or_else(|| bad(C, "no photo"))?;
    let limit = limit_param(p, "limit", 50, C)?;
    let filter = bool_param(p, "filter", false, C)?;
    ensure_open(s);
    let me = s.catalog.photo(id).cloned().ok_or_else(|| bad(C, "no such photo"))?;
    let my_key = content_key(&me);
    // the photo's own digest is made now when the pump has not got to it yet; its vector too, when the lane runs
    let mine = if s.vera.skipped.contains(&my_key) { None } else { digest_now(s, id).ok().flatten() };
    let my_vec = lane::vector_of(s, id, true).ok().flatten();
    let (mut rows, mut ignored, mut pending, mut pending_vec): (Vec<DupRow>, usize, usize, usize) = (Vec::new(), 0, 0, 0);
    for q in s.catalog.photos().filter(|q| eligible(q) && q.id != id) {
        // a virtual copy shows the same file: not a duplicate
        if q.source == me.source {
            continue;
        }
        let key = content_key(q);
        let theirs = s.vera.digests.get(&key).copied();
        let distances = mine.zip(theirs).map(|(a, b)| lightcraft_vera::digest::distances(&a, &b));
        let found = if me.content_hash.is_some() && q.content_hash == me.content_hash {
            Some(("bytes", Some((0, 0)), None))
        } else {
            let by_digest = match (mine, theirs) {
                (Some(a), Some(b)) => match lightcraft_vera::sameness(&a, &b) {
                    Sameness::Same => Some("picture"),
                    Sameness::Possible => Some("possible"),
                    Sameness::Different => None,
                },
                (_, None) => {
                    pending += 1;
                    None
                }
                (None, Some(_)) => None,
            };
            match by_digest {
                Some(tier) => Some((tier, distances, None)),
                // what the digests miss (a crop, a heavier edit): the same picture by the model's eye
                None => {
                    let theirs_vec = s.vera.lane.store.get(&key);
                    if theirs_vec.is_none() {
                        pending_vec += 1;
                    }
                    match (&my_vec, theirs_vec) {
                        (Some(a), Some(b)) => {
                            let c = lightcraft_vera::cutoff::cosine(a, b);
                            (c >= thresholds::EMBED_NEAR).then_some(("similar", distances, Some(c)))
                        }
                        _ => None,
                    }
                }
            }
        };
        let Some((tier, distances, cosine)) = found else { continue };
        if s.vera.ignored.contains(&my_key, &key) {
            ignored += 1;
            continue;
        }
        rows.push(DupRow { tier, id: q.id.0, distances, cosine });
    }
    rows.sort_by(|a, b| {
        tier_rank(a.tier)
            .cmp(&tier_rank(b.tier))
            .then_with(|| b.cosine.unwrap_or(0.0).total_cmp(&a.cosine.unwrap_or(0.0)))
            .then_with(|| a.distances.unwrap_or((u32::MAX, u32::MAX)).cmp(&b.distances.unwrap_or((u32::MAX, u32::MAX))))
            .then(a.id.cmp(&b.id))
    });
    rows.truncate(limit);
    if filter {
        s.filter.only = std::iter::once(id).chain(rows.iter().map(|r| PhotoId(r.id))).collect();
    }
    let photos: Vec<Value> = rows
        .iter()
        .map(|r| {
            let mut row = json!({"id": r.id, "tier": r.tier});
            match r.distances {
                Some((pdq, phash)) => {
                    row["pdq"] = json!(pdq);
                    row["phash"] = json!(phash);
                }
                None => {
                    row["pdq"] = Value::Null;
                    row["phash"] = Value::Null;
                }
            }
            if let Some(c) = r.cosine {
                row["lane"] = json!("vector");
                row["cosine"] = json!(round(c, 4));
                row["strong"] = json!(c >= thresholds::EMBED_DUPLICATE);
            }
            row
        })
        .collect();
    Ok(json!({
        "id": id.0,
        "photos": photos,
        "ignored": ignored,
        "digested": mine.is_some(),
        "pendingDigests": pending,
        "embedded": my_vec.is_some(),
        "pendingEmbeddings": if my_vec.is_some() { pending_vec } else { 0 },
    }))
}

/// The two photos of a pair command, as content keys.
fn pair_keys(s: &Session, p: &Value, c: &str) -> Result<(String, String)> {
    let a = photo_param(s, p, "a", c)?.ok_or_else(|| bad(c, "missing photo `a`"))?;
    let b = photo_param(s, p, "b", c)?.ok_or_else(|| bad(c, "missing photo `b`"))?;
    if a == b {
        return Err(bad(c, "`a` and `b` must be two different photos"));
    }
    let key = |id: PhotoId| s.catalog.photo(id).map(|ph| content_key(ph)).ok_or_else(|| bad(c, format!("no photo {}", id.0)));
    Ok((key(a)?, key(b)?))
}

/// `photo.ignorePair {a, b}`: these two are not the same picture; `photo.duplicates` stops listing them together.
fn ignore_pair(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.ignorePair";
    let (a, b) = pair_keys(s, p, C)?;
    ensure_open(s);
    let added = s.vera.ignored.insert(&a, &b).map_err(|e| bad(C, e))?;
    if added {
        s.vera.ignored.save().map_err(|e| crate::EngineError::Other(format!("could not save the ignored pairs: {e}")))?;
    }
    Ok(json!({"changed": added, "ignoredPairs": s.vera.ignored.len()}))
}

/// `photo.unignorePair {a, b}`: forget a `photo.ignorePair`.
fn unignore_pair(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.unignorePair";
    let (a, b) = pair_keys(s, p, C)?;
    ensure_open(s);
    let removed = s.vera.ignored.remove(&a, &b);
    if removed {
        s.vera.ignored.save().map_err(|e| crate::EngineError::Other(format!("could not save the ignored pairs: {e}")))?;
    }
    Ok(json!({"changed": removed, "ignoredPairs": s.vera.ignored.len()}))
}

/// `photo.stackDuplicate {id, under}`: stack a duplicate under the photo it is a copy of (one undo step).
pub(crate) fn stack_duplicate(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "photo.stackDuplicate";
    let id = photo_param(s, p, "id", C)?.ok_or_else(|| bad(C, "missing `id`"))?;
    let under = photo_param(s, p, "under", C)?.ok_or_else(|| bad(C, "missing `under`"))?;
    if id == under {
        return Err(bad(C, "a photo cannot be stacked under itself"));
    }
    let op = s.catalog.group_ops(under, &[id], true).ok_or_else(|| bad(C, "these photos cannot be stacked"))?;
    s.commit("Stack Duplicate", op)?;
    let st = s.catalog.stack_of(under).map(|st| (st.id.0, st.photos.len())).unwrap_or_default();
    Ok(json!({"stack": st.0, "count": st.1}))
}

pub fn specs() -> Vec<CommandSpec> {
    let mut v = vec![
        cmd!(
            query "vera.status",
            "Search Status",
            [],
            None,
            "{} → {digested, photos, pendingDigests, unreadable, digests, embedded, pendingEmbeddings, unembeddable, embeddings, model: {id, installed, loaded, memoryBytes}, indexDocs, ignoredPairs, lanes: {keyword: \"live\", digest: \"live\", vector: {state: live | needs_model | downloading | disabled_by_user | needs_desktop | error, reason}, describe: {state, reason}}, sidecar: {path?, bytes?}, sidecars: [{name, path?, bytes?}]} — how far fingerprinting and embedding have got, and which search lanes this build runs (with a sentence for each)",
            always,
            status
        ),
        cmd!(
            query "vera.pump",
            "Fingerprint Photos in the Background",
            [],
            None,
            "{pace?: pause | light | normal | full (normal)} → {active, pending, digested, photos, pace, embedding: {state, pending, embedded, photos, inFlight}} — cheap to call every frame: digests the next photos without a fingerprint (none when paused, 1 when light, 2 when normal, 4 when full) and, when the vector lane runs, hands as many photos without an embedding to the embedding thread; keeps the \"Fingerprinting photos\" and \"Embedding photos\" activity rows while there is work and saves both beside the library every few seconds",
            always,
            pump
        ),
        cmd!(
            query "vera.digest",
            "Fingerprint Photos",
            [],
            None,
            "{ids?: [photo ids] (the selection)} → {digested, already, failed: [{id, error}]} — fingerprint these photos now (PDQ + pHash from the thumbnail-level picture), at most 10000",
            always,
            digest
        ),
        cmd!(
            "search.text",
            "Search Photos",
            [],
            None,
            "{query, limit?: 1..10000 (200), filter?: bool (true)} → {photos: [{id, score, lane: keyword | vector | both, cosine?, keywordScore?}], cutoff, lanes, filtered, warning?} — find photos by words in their title, keywords (every level), caption, alt text, description, file name, camera, lens and place (BM25, each word a prefix, any word may match) and, when the vector lane runs (lanes: [\"keyword\", \"vector\"]), by meaning: the two ranked lists fused by reciprocal rank (score), vector-only matches under the noise ceiling (cutoff) left out, keyword matches never. Best first; with `filter`, the view shows the results in rank order (sort: relevance; Clear Filters to go back). A search that finds nothing leaves the view alone; `warning` says when the vector lane should have run and could not",
            always,
            search_text
        ),
        cmd!(
            "photo.duplicates",
            "Find Duplicates",
            [],
            None,
            "{id?: photo (the active one), limit?: 1..10000 (50), filter?: bool (false)} → {id, photos: [{id, tier: bytes | picture | possible | similar, pdq, phash, lane?: \"vector\", cosine?, strong?}], ignored, digested, pendingDigests, embedded, pendingEmbeddings} — other photos that are the same picture: the same bytes, the same picture by fingerprint (a resize, a re-export), possibly the same (a variant edit, a burst neighbour), or, by the vector lane where the fingerprints do not match, similar (a crop, a heavier edit: cosine at or over the near-duplicate bar, `strong` at or over the duplicate bar); `pdq` / `phash` are the bits that differ (of 256 / 64; null without both fingerprints). Virtual copies and pairs set apart with photo.ignorePair are left out. With `filter`, the view shows the photo and its duplicates",
            always,
            duplicates
        ),
        cmd!(
            "photo.similar",
            "Find Photos That Look Alike",
            [],
            None,
            "{id?: photo (the active one), limit?: 1..10000 (50), filter?: bool (true)} → {id, photos: [{id, cosine, lane: \"vector\"}], cutoff, embedded, pendingEmbeddings, filtered} — the photos that look most like this one by the vector lane (cosine of their SigLIP 2 embeddings), best first, those under the noise ceiling (cutoff) left out. The photo is embedded on the spot if the background pass has not got to it; an error says when the vector lane is not ready. With `filter`, the view shows the photo and the results in rank order (sort: relevance)",
            always,
            similar
        ),
        cmd!(
            "photo.ignorePair",
            "Ignore Duplicate Pair",
            [],
            None,
            "{a: photo id, b: photo id} → {changed, ignoredPairs} — these two are not the same picture: photo.duplicates and the import check stop pairing them (kept beside the library, not an undo step)",
            always,
            ignore_pair
        ),
        cmd!(
            "photo.unignorePair",
            "Unignore Duplicate Pair",
            [],
            None,
            "{a: photo id, b: photo id} → {changed, ignoredPairs} — forget a photo.ignorePair",
            always,
            unignore_pair
        ),
        cmd!(
            "photo.stackDuplicate",
            "Stack Duplicate",
            [],
            None,
            "{id: photo id, under: photo id} → {stack, count} — stack a duplicate under the photo it copies (collapsed, `under` on top; stacks either is in are merged); one undo step",
            always,
            stack_duplicate
        ),
    ];
    v.extend(super::model::specs());
    v
}
