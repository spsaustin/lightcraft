//! VERA's commands: `vera.status`, `vera.pump`, `vera.digest`, `search.text`, `photo.duplicates`,
//! `photo.ignorePair` / `photo.unignorePair` and `photo.stackDuplicate` (`docs/vera.md` → Commands).
//!
//! Every parameter is hostile input: types are checked, lists and limits are capped at [`MAX_LIMIT`].

use lightcraft_catalog::{PhotoId, Sort, SortKey};
use lightcraft_vera::Sameness;
use serde_json::{Value, json};

use super::{Pace, digest_now, eligible, ensure_index, ensure_open};
use crate::cmd::{CommandSpec, always, bad, cmd};
use crate::media::content_key;
use crate::{Result, Session};

/// The most any list parameter or limit may ask for.
pub(crate) const MAX_LIMIT: usize = 10_000;
/// Longest search query.
const MAX_QUERY: usize = 2000;

/// Why the meaning lane is not running, said the same way everywhere.
const VECTOR_REASON: &str = "Finding photos by meaning needs an image-and-text embedding model, which this build does not include yet (Phase 2).";
const DESCRIBE_REASON: &str = "Describing photos in words needs a captioning model; it is not part of this build (Phase 4).";

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
    Ok(json!({
        "digested": digested,
        "photos": photos,
        "pendingDigests": photos.saturating_sub(digested).saturating_sub(skipped),
        "unreadable": skipped,
        "digests": s.vera.digests.len(),
        "indexDocs": s.vera.index.as_ref().map_or(0, |i| i.len()),
        "ignoredPairs": s.vera.ignored.len(),
        "lanes": {
            "keyword": "live",
            "digest": "live",
            "vector": {"state": "needs_model", "reason": VECTOR_REASON},
            "describe": {"state": "disabled", "reason": DESCRIBE_REASON},
        },
        "sidecar": sidecar,
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

/// `search.text {query, limit?, filter?}`: the keyword lane, ranked.
fn search_text(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "search.text";
    let query = match p.get("query") {
        Some(Value::String(q)) => q.as_str(),
        Some(_) => return Err(bad(C, "`query` must be text")),
        None => return Err(bad(C, "missing `query`")),
    };
    if query.chars().count() > MAX_QUERY {
        return Err(bad(C, format!("a query is at most {MAX_QUERY} characters")));
    }
    let limit = limit_param(p, "limit", 200, C)?;
    let filter = bool_param(p, "filter", true, C)?;
    ensure_index(s);
    let hits = s.vera.index.as_ref().map(|i| i.search(query, limit)).unwrap_or_default();
    // a search that found nothing leaves the view alone: an empty `only` would show everything
    let filtered = filter && !hits.is_empty();
    if filtered {
        s.filter.only = hits.iter().map(|h| PhotoId(h.doc)).collect();
        s.sort = Sort { key: SortKey::Relevance, ascending: false, ..s.sort };
    }
    let photos: Vec<Value> = hits.iter().map(|h| json!({"id": h.doc, "score": (h.score * 1000.0).round() / 1000.0, "lane": "keyword"})).collect();
    Ok(json!({"photos": photos, "cutoff": null, "lanes": ["keyword"], "filtered": filtered}))
}

/// Order of the duplicate tiers in a list.
fn tier_rank(tier: &str) -> u8 {
    match tier {
        "bytes" => 0,
        "picture" => 1,
        _ => 2,
    }
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
    // the photo's own digest is made now when the pump has not got to it yet
    let mine = if s.vera.skipped.contains(&my_key) { None } else { digest_now(s, id).ok().flatten() };
    let (mut rows, mut ignored, mut pending): (Vec<(u8, u32, u32, u64, &'static str)>, usize, usize) = (Vec::new(), 0, 0);
    for q in s.catalog.photos().filter(|q| eligible(q) && q.id != id) {
        // a virtual copy shows the same file: not a duplicate
        if q.source == me.source {
            continue;
        }
        let key = content_key(q);
        let found = if me.content_hash.is_some() && q.content_hash == me.content_hash {
            Some(("bytes", 0, 0))
        } else {
            match (mine, s.vera.digests.get(&key)) {
                (Some(a), Some(b)) => {
                    let (pdq, phash) = lightcraft_vera::digest::distances(&a, b);
                    match lightcraft_vera::sameness(&a, b) {
                        Sameness::Same => Some(("picture", pdq, phash)),
                        Sameness::Possible => Some(("possible", pdq, phash)),
                        Sameness::Different => None,
                    }
                }
                (_, None) => {
                    pending += 1;
                    None
                }
                (None, Some(_)) => None,
            }
        };
        let Some((tier, pdq, phash)) = found else { continue };
        if s.vera.ignored.contains(&my_key, &key) {
            ignored += 1;
            continue;
        }
        rows.push((tier_rank(tier), pdq, phash, q.id.0, tier));
    }
    rows.sort();
    rows.truncate(limit);
    if filter {
        s.filter.only = std::iter::once(id).chain(rows.iter().map(|r| PhotoId(r.3))).collect();
    }
    let photos: Vec<Value> = rows.iter().map(|(_, pdq, phash, q, tier)| json!({"id": q, "tier": tier, "pdq": pdq, "phash": phash})).collect();
    Ok(json!({"id": id.0, "photos": photos, "ignored": ignored, "digested": mine.is_some(), "pendingDigests": pending}))
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
    vec![
        cmd!(
            query "vera.status",
            "Search Status",
            [],
            None,
            "{} → {digested, photos, pendingDigests, unreadable, digests, indexDocs, ignoredPairs, lanes: {keyword: \"live\", digest: \"live\", vector: {state, reason}, describe: {state, reason}}, sidecar: {path?, bytes?}} — how far fingerprinting has got, and which search lanes this build runs",
            always,
            status
        ),
        cmd!(
            query "vera.pump",
            "Fingerprint Photos in the Background",
            [],
            None,
            "{pace?: pause | light | normal | full (normal)} → {active, pending, digested, photos, pace} — cheap to call every frame: digests the next photos without a fingerprint (none when paused, 1 when light, 2 when normal, 4 when full), keeps the \"Fingerprinting photos\" activity row while there is work and saves the digests beside the library every few seconds",
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
            "{query, limit?: 1..10000 (200), filter?: bool (true)} → {photos: [{id, score, lane: \"keyword\"}], cutoff: null, lanes: [\"keyword\"], filtered} — find photos by words in their title, keywords (every level), caption, alt text, description, file name, camera, lens and place, best first (BM25, each word a prefix, any word may match); with `filter`, the view shows the results in rank order (sort: relevance; Clear Filters to go back). A search that finds nothing leaves the view alone",
            always,
            search_text
        ),
        cmd!(
            "photo.duplicates",
            "Find Duplicates",
            [],
            None,
            "{id?: photo (the active one), limit?: 1..10000 (50), filter?: bool (false)} → {id, photos: [{id, tier: bytes | picture | possible, pdq, phash}], ignored, digested, pendingDigests} — other photos that are the same picture: the same bytes, the same picture by fingerprint (a resize, a re-export) or possibly the same (a variant edit, a burst neighbour); `pdq` / `phash` are the bits that differ (of 256 / 64). Virtual copies and pairs set apart with photo.ignorePair are left out. With `filter`, the view shows the photo and its duplicates",
            always,
            duplicates
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
    ]
}
