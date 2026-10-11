//! VERA's commands on a session. Digests are injected where a test needs two pictures to match, so nothing here
//! depends on what the digest algorithm makes of the demo scenes.

use lightcraft_catalog::{PhotoId, SortKey};
use lightcraft_vera::Digest;
use serde_json::{Value, json};

use crate::Session;
use crate::media::content_key;

fn demo() -> Session {
    Session::with_demo()
}

fn key_of(s: &Session, id: PhotoId) -> String {
    content_key(s.catalog.photo(id).unwrap())
}

/// Two photos of the demo library that show different files (not virtual copies of one another), and a third.
fn three(s: &Session) -> [PhotoId; 3] {
    let mut picked: Vec<&lightcraft_catalog::Photo> = Vec::new();
    for p in s.catalog.photos().filter(|p| super::eligible(p)) {
        if picked.iter().all(|q| q.source != p.source && content_key(q) != content_key(p)) {
            picked.push(p);
        }
        if picked.len() == 3 {
            break;
        }
    }
    assert_eq!(picked.len(), 3, "the demo library has three distinct pictures");
    [picked[0].id, picked[1].id, picked[2].id]
}

fn digest(pdq: u8, phash: u64) -> Digest {
    Digest { pdq: [pdq; 32], pdq_quality: 80, phash }
}

#[test]
fn vera_status_reports_lanes_and_counts() {
    let mut s = demo();
    let r = s.execute("vera.status", &json!({})).unwrap();
    let photos = s.catalog.photos().filter(|p| super::eligible(p)).count();
    assert!(photos > 0);
    assert_eq!(r["photos"], photos, "{r}");
    assert_eq!(r["digested"], 0, "{r}");
    assert_eq!(r["pendingDigests"], photos, "{r}");
    assert_eq!(r["lanes"]["keyword"], "live");
    assert_eq!(r["lanes"]["digest"], "live");
    assert_eq!(r["lanes"]["vector"]["state"], "needs_model");
    assert!(r["lanes"]["vector"]["reason"].as_str().is_some_and(|t| !t.is_empty()));
    assert_eq!(r["lanes"]["describe"]["state"], "needs_endpoint");
    // an in-memory library keeps its digests in memory only
    assert_eq!(r["sidecar"], json!({}));
    assert!(r["indexDocs"].is_u64());
    // a status call is not journaled
    assert!(!s.journal.iter().any(|(id, _)| id == "vera.status"));
}

#[test]
fn vera_pump_digests_the_library_with_an_activity_row() {
    let mut s = demo();
    let photos = s.catalog.photos().filter(|p| super::eligible(p)).count();
    // paused: nothing is done
    let r = s.execute("vera.pump", &json!({"pace": "pause"})).unwrap();
    assert_eq!(r["digested"], 0, "{r}");
    assert!(s.vera.digests.is_empty());
    assert!(s.execute("vera.pump", &json!({"pace": "flat out"})).is_err());
    assert!(s.execute("vera.pump", &json!({"pace": 3})).is_err());

    let first = s.execute("vera.pump", &json!({"pace": "light"})).unwrap();
    assert!(first["digested"].as_u64().is_some_and(|n| n >= 1), "{first}");
    if photos > 1 {
        assert_eq!(first["active"], true, "{first}");
        let rows = s.activity.list();
        let row = rows.iter().find(|t| t.kind == "vera").expect("an activity row while there is work");
        assert_eq!(row.label, "Fingerprinting photos");
        assert!(!row.cancellable);
        assert!(row.total >= 1 && row.done == 0, "the whole is what was pending at most: {row:?}");
    }
    let mut last = first;
    for _ in 0..(photos * 2 + 10) {
        if last["active"] == false {
            break;
        }
        last = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
    }
    assert_eq!(last["active"], false, "{last}");
    assert_eq!(last["pending"], 0, "{last}");
    assert_eq!(last["digested"], photos, "{last}");
    assert_eq!(last["photos"], photos, "{last}");
    assert!(!s.activity.list().iter().any(|t| t.kind == "vera"), "the row is gone once the work is done");
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["digested"].as_u64(), st["pendingDigests"].as_u64()), (Some(photos as u64), Some(0)), "{st}");
    // nothing left: a further call does nothing and stays cheap
    let again = s.execute("vera.pump", &json!({})).unwrap();
    assert_eq!(again["active"], false);
    assert!(!s.journal.iter().any(|(id, _)| id == "vera.pump"), "the per-frame pump is not journaled");
}

#[test]
fn vera_digest_does_named_photos_now() {
    let mut s = demo();
    let [a, b, _] = three(&s);
    let r = s.execute("vera.digest", &json!({"ids": [a.0, b.0, 999_999]})).unwrap();
    assert_eq!(r["digested"], 2, "{r}");
    assert_eq!(r["failed"][0]["id"], 999_999, "{r}");
    assert!(s.vera.digests.contains(&key_of(&s, a)));
    let r = s.execute("vera.digest", &json!({"ids": [a.0]})).unwrap();
    assert_eq!((r["digested"].as_u64(), r["already"].as_u64()), (Some(0), Some(1)));
    assert!(s.execute("vera.digest", &json!({"ids": "all"})).is_err());
    assert!(s.execute("vera.digest", &json!({"ids": ["x"]})).is_err());
    let many: Vec<u64> = (0..10_001).collect();
    assert!(s.execute("vera.digest", &json!({"ids": many})).is_err());
}

#[test]
fn vera_search_text_ranks_and_filters_the_view() {
    let mut s = demo();
    let [a, _, _] = three(&s);
    s.execute("library.select", &json!({"ids": [a.0], "active": a.0})).unwrap();
    s.execute("photo.setMeta", &json!({"ids": [a.0], "title": "Golden hour surf"})).unwrap();
    let r = s.execute("search.text", &json!({"query": "golden SURF"})).unwrap();
    assert_eq!(r["photos"][0]["id"], a.0, "{r}");
    assert_eq!(r["photos"][0]["lane"], "keyword");
    assert_eq!(r["lanes"], json!(["keyword"]));
    assert_eq!(r["cutoff"], Value::Null);
    assert_eq!(r["filtered"], true);
    assert_eq!(s.filter.only.first(), Some(&a));
    assert_eq!(s.sort.key, SortKey::Relevance);
    assert_eq!(s.catalog.query(&s.filter, &s.sort).first(), Some(&a), "the grid shows the best match first");
    assert_eq!(s.visible().first(), Some(&a));
    // a prefix finds it too; the index follows metadata changes
    let r = s.execute("search.text", &json!({"query": "surfi", "filter": false})).unwrap();
    assert!(r["photos"].as_array().unwrap().is_empty(), "{r}");
    let r = s.execute("search.text", &json!({"query": "gold", "filter": false})).unwrap();
    assert!(r["photos"].as_array().unwrap().iter().any(|h| h["id"] == a.0), "{r}");
    s.execute("photo.setMeta", &json!({"ids": [a.0], "title": "Blue morning"})).unwrap();
    let r = s.execute("search.text", &json!({"query": "golden", "filter": false})).unwrap();
    assert!(!r["photos"].as_array().unwrap().iter().any(|h| h["id"] == a.0), "{r}");
    // nothing found: the view is left alone
    let before = s.filter.clone();
    let r = s.execute("search.text", &json!({"query": "qqqzzzxx"})).unwrap();
    assert_eq!((r["photos"].as_array().map(Vec::len), r["filtered"].as_bool()), (Some(0), Some(false)));
    assert_eq!(s.filter, before);
    // hostile parameters
    assert!(s.execute("search.text", &json!({})).is_err());
    assert!(s.execute("search.text", &json!({"query": 5})).is_err());
    assert!(s.execute("search.text", &json!({"query": "x", "limit": "all"})).is_err());
    assert!(s.execute("search.text", &json!({"query": "x", "filter": "yes"})).is_err());
    assert!(s.execute("search.text", &json!({"query": "x".repeat(5000)})).is_err());
    assert!(s.execute("search.text", &json!({"query": "golden", "limit": u64::MAX})).is_ok());
}

#[test]
fn vera_search_text_indexes_keywords_file_names_and_places() {
    let mut s = demo();
    let [a, b, _] = three(&s);
    s.execute("library.select", &json!({"ids": [a.0, b.0], "active": a.0})).unwrap();
    s.execute("photo.setMeta", &json!({"ids": [a.0], "keywords": ["travel|Portugal|Nazaré"]})).unwrap();
    s.execute("photo.setMeta", &json!({"ids": [b.0], "city": "Lisbon"})).unwrap();
    let r = s.execute("search.text", &json!({"query": "portugal", "filter": false})).unwrap();
    assert_eq!(r["photos"][0]["id"], a.0, "{r}");
    let r = s.execute("search.text", &json!({"query": "nazaré", "filter": false})).unwrap();
    assert_eq!(r["photos"][0]["id"], a.0, "{r}");
    let r = s.execute("search.text", &json!({"query": "lisbon", "filter": false})).unwrap();
    assert_eq!(r["photos"][0]["id"], b.0, "{r}");
    let stem = std::path::Path::new(&s.catalog.photo(a).unwrap().file_name).file_stem().unwrap().to_string_lossy().to_string();
    let r = s.execute("search.text", &json!({"query": stem, "filter": false, "limit": 10_000})).unwrap();
    assert!(r["photos"].as_array().unwrap().iter().any(|h| h["id"] == a.0), "{r}");
    assert_eq!(s.execute("vera.status", &json!({})).unwrap()["indexDocs"], s.catalog.photos().filter(|p| p.in_library()).count());
}

#[test]
fn vera_photo_duplicates_tiers_and_ignored_pairs() {
    let mut s = demo();
    let [a, b, c] = three(&s);
    let base = digest(0xAA, 0x0123_4567_89AB_CDEF);
    let mut near = base;
    // 40 PDQ bits and 10 pHash bits apart: possibly the same picture, not the same
    for byte in near.pdq.iter_mut().take(5) {
        *byte ^= 0xFF;
    }
    near.phash ^= 0x3FF;
    let (ka, kb, kc) = (key_of(&s, a), key_of(&s, b), key_of(&s, c));
    s.vera.digests.insert(ka, base);
    s.vera.digests.insert(kb, base);
    s.vera.digests.insert(kc, near);
    let r = s.execute("photo.duplicates", &json!({"id": a.0})).unwrap();
    let rows = r["photos"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "{r}");
    assert_eq!(rows[0], json!({"id": b.0, "tier": "picture", "pdq": 0, "phash": 0}), "{r}");
    assert_eq!(rows[1], json!({"id": c.0, "tier": "possible", "pdq": 40, "phash": 10}), "{r}");
    assert_eq!(r["ignored"], 0);
    assert_eq!(r["digested"], true);
    assert!(s.filter.only.is_empty(), "the view is not filtered by default");

    // b is not the same picture as a after all
    let r = s.execute("photo.ignorePair", &json!({"a": b.0, "b": a.0})).unwrap();
    assert_eq!((r["changed"].as_bool(), r["ignoredPairs"].as_u64()), (Some(true), Some(1)));
    assert_eq!(s.execute("photo.ignorePair", &json!({"a": a.0, "b": b.0})).unwrap()["changed"], false, "(a, b) is (b, a)");
    let r = s.execute("photo.duplicates", &json!({"id": a.0, "filter": true})).unwrap();
    let rows = r["photos"].as_array().unwrap();
    assert_eq!((rows.len(), r["ignored"].as_u64()), (1, Some(1)), "{r}");
    assert_eq!(rows[0]["id"], c.0);
    assert_eq!(s.filter.only, vec![a, c], "filtered: the photo and its duplicates");
    s.execute("photo.unignorePair", &json!({"a": a.0, "b": b.0})).unwrap();
    let r = s.execute("photo.duplicates", &json!({"id": a.0, "limit": 1})).unwrap();
    assert_eq!(r["photos"].as_array().map(Vec::len), Some(1), "{r}");
    assert_eq!(r["photos"][0]["id"], b.0);

    // hostile parameters
    assert!(s.execute("photo.duplicates", &json!({"id": "a"})).is_err());
    assert!(s.execute("photo.duplicates", &json!({"id": 999_999})).is_err());
    assert!(s.execute("photo.ignorePair", &json!({"a": a.0, "b": a.0})).is_err());
    assert!(s.execute("photo.ignorePair", &json!({"a": a.0})).is_err());
    assert!(s.execute("photo.unignorePair", &json!({"a": a.0, "b": -1})).is_err());
}

#[test]
fn vera_stack_duplicate_is_one_undo_step() {
    let mut s = demo();
    let [a, b, _] = three(&s);
    let r = s.execute("photo.stackDuplicate", &json!({"id": b.0, "under": a.0})).unwrap();
    assert_eq!(r["count"], 2, "{r}");
    let st = s.catalog.stack_of(a).unwrap();
    assert_eq!(st.photos, vec![a, b]);
    assert!(st.collapsed);
    assert_eq!(s.undo.last().map(|u| u.label.as_str()), Some("Stack Duplicate"));
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.catalog.stack_of(a).is_none());
    assert!(s.execute("photo.stackDuplicate", &json!({"id": a.0, "under": a.0})).is_err());
    assert!(s.execute("photo.stackDuplicate", &json!({"id": a.0})).is_err());
    assert!(s.execute("photo.stackDuplicate", &json!({"id": a.0, "under": 999_999})).is_err());
}

#[test]
fn vera_sidecars_follow_the_library_folder() {
    let dir = std::env::temp_dir().join(format!("lc-vera-lib-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut s = Session::new();
    s.open_library(&dir, true).unwrap();
    let [a, b, _] = three(&s);
    s.execute("vera.digest", &json!({"ids": [a.0, b.0]})).unwrap();
    s.execute("photo.ignorePair", &json!({"a": a.0, "b": b.0})).unwrap();
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert!(st["sidecar"]["path"].as_str().is_some_and(|p| p.ends_with(super::digests::FILE_NAME)), "{st}");
    assert!(st["sidecar"]["bytes"].as_u64().is_some_and(|n| n > 5), "{st}");
    assert!(dir.join(super::IGNORED_FILE).is_file());
    s.close_library().unwrap();
    drop(s);
    // a new session over the same library reads both back
    let mut s = Session::new();
    s.open_library(&dir, false).unwrap();
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["digested"].as_u64(), st["ignoredPairs"].as_u64()), (Some(2), Some(1)), "{st}");
    // another (in-memory) library starts with nothing
    s.open_library_in(
        crate::library::LibraryStores {
            dir: "memory".into(),
            catalog: Box::new(lightcraft_catalog::MemStore::default()),
            files: Box::new(lightcraft_catalog::MemStore::default()),
            on_disk: false,
        },
        true,
    )
    .unwrap();
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["digested"].as_u64(), st["ignoredPairs"].as_u64(), st["sidecar"].clone()), (Some(0), Some(0), json!({})), "{st}");
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}
