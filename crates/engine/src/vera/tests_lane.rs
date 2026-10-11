//! The vector lane on a session. Vectors are put in the store by hand and query embeddings come from
//! [`super::lane::Lane::test_queries`], so nothing here needs the model, except the last test, which runs the real pump
//! on the demo library when `LIGHTCRAFT_SIGLIP2_DIR` names a checkpoint.

use std::path::{Path, PathBuf};

use lightcraft_catalog::{PhotoId, SortKey};
use lightcraft_embed::DIM;
use lightcraft_vera::Digest;
use lightcraft_vera::cutoff::mix64;
use serde_json::{Value, json};

use crate::Session;
use crate::media::content_key;

fn key_of(s: &Session, id: PhotoId) -> String {
    content_key(s.catalog.photo(id).unwrap())
}

/// One photo per distinct picture of the library (no virtual copies), in id order.
fn distinct(s: &Session) -> Vec<PhotoId> {
    let mut picked: Vec<&lightcraft_catalog::Photo> = Vec::new();
    for p in s.catalog.photos().filter(|p| super::eligible(p)) {
        if picked.iter().all(|q| q.source != p.source && content_key(q) != content_key(p)) {
            picked.push(p);
        }
    }
    let mut ids: Vec<PhotoId> = picked.iter().map(|p| p.id).collect();
    ids.sort();
    ids
}

/// A deterministic pseudo-random unit vector (random 768-d vectors are nearly orthogonal: cosines about ±0.04).
fn random_vec(seed: u64) -> Vec<f32> {
    let v: Vec<f32> = (0..DIM as u64).map(|i| (mix64(seed.wrapping_mul(0x9E37_79B9) ^ i) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32).collect();
    unit(&v)
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

/// `a·wa + b'·wb` for unit `a` and `b'` (`b` with its part along `a` taken out): its cosine with `a` is exactly `wa`
/// when `wa² + wb² = 1`.
fn mix(a: &[f32], wa: f32, b: &[f32], wb: f32) -> Vec<f32> {
    let along: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ortho = unit(&b.iter().zip(a).map(|(y, x)| y - along * x).collect::<Vec<_>>());
    unit(&a.iter().zip(&ortho).map(|(x, y)| x * wa + y * wb).collect::<Vec<_>>())
}

/// Every distinct picture of the demo library gets a random vector; returns them by photo.
fn embed_all(s: &mut Session) -> Vec<(PhotoId, Vec<f32>)> {
    let ids = distinct(s);
    let mut out = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let v = random_vec(1000 + i as u64);
        let key = key_of(s, *id);
        assert!(s.vera.lane.store.insert(key, &v));
        out.push((*id, v));
    }
    out
}

fn rows(r: &Value) -> &Vec<Value> {
    r["photos"].as_array().unwrap()
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lc-vera-lane-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A folder that looks like a model to `is_model_dir` but holds no real checkpoint.
fn fake_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    for f in [lightcraft_embed::WEIGHTS_FILE, lightcraft_embed::CONFIG_FILE, lightcraft_embed::TOKENIZER_FILE] {
        std::fs::write(dir.join(f), b"not a model").unwrap();
    }
}

#[test]
fn vera_status_reports_the_vector_lane() {
    let mut s = Session::with_demo();
    let r = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(r["lanes"]["vector"]["state"], "needs_model", "{r}");
    assert!(r["lanes"]["vector"]["reason"].as_str().unwrap().contains("SigLIP 2"), "{r}");
    assert_eq!(r["model"]["id"], super::lane::MODEL_ID);
    assert_eq!(r["model"]["installed"], false);
    assert_eq!(r["embedded"], 0);
    assert_eq!(r["pendingEmbeddings"], r["photos"], "{r}");
    let sidecars = r["sidecars"].as_array().unwrap();
    assert_eq!(sidecars.len(), 2, "{r}");
    assert_eq!(sidecars[0]["name"], "vera-digests.bin");
    assert_eq!(sidecars[1]["name"], format!("vera-embeddings-{}.bin", super::lane::MODEL_ID));
    // with a model folder that is empty, the reason says where the files go
    let dir = temp_dir("status");
    s.vera_model_dir = Some(dir.join("siglip2"));
    let r = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(r["lanes"]["vector"]["state"], "needs_model");
    assert!(r["lanes"]["vector"]["reason"].as_str().unwrap().contains("siglip2"), "{r}");
    // the pump has nothing to do without a model, and says so
    let p = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
    assert_eq!(p["embedding"]["state"], "needs_model", "{p}");
    assert_eq!(p["embedding"]["pending"], 0, "{p}");
    assert!(!s.activity.list().iter().any(|t| t.label == "Embedding photos"));
    // once the files are there it is live; switched off for the library it says so
    fake_model(&dir.join("siglip2"));
    let r = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((r["lanes"]["vector"]["state"].as_str(), r["model"]["installed"].as_bool()), (Some("live"), Some(true)), "{r}");
    s.execute("vera.config", &json!({"vectorLane": false})).unwrap();
    let r = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(r["lanes"]["vector"]["state"], "disabled_by_user", "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vera_a_damaged_model_is_an_error_state_not_a_crash() {
    let mut s = Session::with_demo();
    let dir = temp_dir("damaged");
    fake_model(&dir);
    s.vera_model_dir = Some(dir.clone());
    let p = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
    assert_eq!(p["embedding"]["state"], "error", "{p}");
    let r = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(r["lanes"]["vector"]["state"], "error", "{r}");
    assert!(r["lanes"]["vector"]["reason"].as_str().unwrap().contains("could not be used"), "{r}");
    // a search falls back to the keyword lane and says why
    let r = s.execute("search.text", &json!({"query": "lake", "filter": false})).unwrap();
    assert_eq!(r["lanes"], json!(["keyword"]), "{r}");
    assert!(r["warning"].as_str().is_some_and(|w| w.contains("keyword matches only")), "{r}");
    assert!(!rows(&r).is_empty(), "the keyword lane still answers: {r}");
    // photo.similar says the lane is not ready
    let id = distinct(&s)[0];
    let e = s.execute("photo.similar", &json!({"id": id.0})).unwrap_err().to_string();
    assert!(e.contains("not ready"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vera_search_text_fuses_the_keyword_and_vector_lanes() {
    let mut s = Session::with_demo();
    let vecs = embed_all(&mut s);
    assert!(vecs.len() >= 8, "enough pictures for the cutoff to engage");
    // a library-sized population of unrelated pictures (keys no photo has: they shape the noise, never show)
    for i in 0..1000 {
        assert!(s.vera.lane.store.insert(format!("noise-{i}"), &random_vec(50_000 + i)));
    }
    let (a, b, c) = (vecs[0].0, vecs[1].0, vecs[2].0);
    s.execute("library.select", &json!({"ids": [a.0, c.0], "active": a.0})).unwrap();
    s.execute("photo.setMeta", &json!({"ids": [a.0, c.0], "title": "Zebra crossing"})).unwrap();
    // the query lies between b (cosine 0.8) and a (0.6); c matches only by its words
    let q = mix(&vecs[1].1, 0.8, &vecs[0].1, 0.6);
    assert!((lightcraft_vera::cutoff::cosine(&q, &vecs[0].1) - 0.6).abs() < 0.05);
    s.vera.lane.test_queries.insert("zebra".into(), q);
    let r = s.execute("search.text", &json!({"query": "zebra"})).unwrap();
    assert_eq!(r["lanes"], json!(["keyword", "vector"]), "{r}");
    let cutoff = r["cutoff"].as_f64().expect("the noise ceiling engaged");
    assert!(cutoff > 0.1 && cutoff < 0.5, "{r}");
    let list = rows(&r);
    let row = |id: PhotoId| list.iter().find(|h| h["id"] == id.0).cloned();
    // a is found by both lanes and comes first; b only by meaning; c only by its words, and never cut
    assert_eq!(list[0]["id"], a.0, "{r}");
    assert_eq!(row(a).unwrap()["lane"], "both");
    assert_eq!(row(b).unwrap()["lane"], "vector");
    assert_eq!(row(c).unwrap()["lane"], "keyword");
    assert!((row(b).unwrap()["cosine"].as_f64().unwrap() - 0.8).abs() < 0.05, "{r}");
    assert!(row(c).unwrap()["cosine"].is_null(), "under the ceiling: {r}");
    // nothing else got past the ceiling: random vectors are noise
    assert_eq!(list.len(), 3, "{r}");
    let scores: Vec<f64> = list.iter().map(|h| h["score"].as_f64().unwrap()).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "fused best first: {scores:?}");
    // reciprocal rank fusion: a has a share from each lane (1/(60 + rank) twice), b and c one each
    assert!(scores[0] >= 2.0 / 62.0 - 1e-5 && scores[1] <= 1.0 / 61.0 + 1e-5, "{scores:?}");
    // the view shows them in fused order
    assert_eq!(r["filtered"], true);
    assert_eq!(s.filter.only.first(), Some(&a));
    assert_eq!(s.sort.key, SortKey::Relevance);
    // a query whose vector matches nothing is the keyword lane alone, ceiling still reported
    s.vera.lane.test_queries.insert("crossing".into(), random_vec(424_242));
    let r = s.execute("search.text", &json!({"query": "crossing", "filter": false})).unwrap();
    assert!(rows(&r).iter().all(|h| h["lane"] == "keyword"), "{r}");
    assert_eq!(rows(&r).len(), 2, "{r}");
    // without a query vector the Phase 1 shape is unchanged
    let r = s.execute("search.text", &json!({"query": "zebr", "filter": false})).unwrap();
    assert_eq!((r["lanes"].clone(), r["cutoff"].clone()), (json!(["keyword"]), Value::Null), "{r}");
}

#[test]
fn vera_photo_similar_ranks_by_cosine_above_the_ceiling() {
    let mut s = Session::with_demo();
    let vecs = embed_all(&mut s);
    let (a, b, c) = (vecs[0].0, vecs[1].0, vecs[2].0);
    // b and c look like a (cosines 0.95 and 0.7); the rest are noise. The test vectors are near-orthogonal, so this
    // query's ceiling is low and both clear it; the cap at EMBED_NEAR only ever lowers a ceiling.
    let vb = mix(&vecs[0].1, 0.95, &vecs[1].1, (1.0f32 - 0.9025).sqrt());
    let vc = mix(&vecs[0].1, 0.7, &vecs[2].1, (1.0f32 - 0.49).sqrt());
    s.vera.lane.store.insert(key_of(&s, b), &vb);
    s.vera.lane.store.insert(key_of(&s, c), &vc);
    let r = s.execute("photo.similar", &json!({"id": a.0})).unwrap();
    let list = rows(&r);
    assert_eq!(list.len(), 2, "only the two above the noise ceiling: {r}");
    assert_eq!((list[0]["id"].as_u64(), list[1]["id"].as_u64()), (Some(b.0), Some(c.0)), "{r}");
    assert!((list[0]["cosine"].as_f64().unwrap() - 0.95).abs() < 0.05, "{r}");
    assert!(r["cutoff"].as_f64().unwrap() <= f64::from(lightcraft_vera::thresholds::EMBED_NEAR) + 1e-6, "capped at EMBED_NEAR: {r}");
    assert!(list.iter().all(|h| h["lane"] == "vector"));
    assert!(r["cutoff"].as_f64().is_some(), "{r}");
    assert_eq!((r["id"].as_u64(), r["embedded"].as_bool(), r["pendingEmbeddings"].as_u64()), (Some(a.0), Some(true), Some(0)), "{r}");
    // the view shows the photo then its look-alikes, in rank order
    assert_eq!(s.filter.only, vec![a, b, c]);
    assert_eq!(s.sort.key, SortKey::Relevance);
    // limit, no filter
    let before = s.filter.clone();
    let r = s.execute("photo.similar", &json!({"id": a.0, "limit": 1, "filter": false})).unwrap();
    assert_eq!(rows(&r).len(), 1);
    assert_eq!(s.filter, before);
    // a photo with no vector and no model: the lane is not ready
    let mut fresh = Session::with_demo();
    let e = fresh.execute("photo.similar", &json!({"id": a.0})).unwrap_err().to_string();
    assert!(e.contains("not ready") && e.contains("SigLIP 2"), "{e}");
    // hostile parameters
    assert!(s.execute("photo.similar", &json!({"id": "a"})).is_err());
    assert!(s.execute("photo.similar", &json!({"id": 999_999})).is_err());
    assert!(s.execute("photo.similar", &json!({"id": a.0, "limit": -3})).is_err());
    assert!(s.execute("photo.similar", &json!({"id": a.0, "filter": 1})).is_err());
}

#[test]
fn vera_photo_duplicates_has_a_similar_tier_for_what_digests_miss() {
    let mut s = Session::with_demo();
    let vecs = embed_all(&mut s);
    let (a, b, c, d) = (vecs[0].0, vecs[1].0, vecs[2].0, vecs[3].0);
    // a and b: a crop (fingerprints far apart) the model sees as the same picture (cosine 0.96 ≥ EMBED_DUPLICATE);
    // c: near (0.92, between EMBED_NEAR and EMBED_DUPLICATE); d: 0.5, not listed
    let far_a = Digest { pdq: [0x00; 32], pdq_quality: 90, phash: 0 };
    let far_b = Digest { pdq: [0xFF; 32], pdq_quality: 90, phash: u64::MAX };
    s.vera.digests.insert(key_of(&s, a), far_a);
    s.vera.digests.insert(key_of(&s, b), far_b);
    for (id, w, own) in [(b, 0.96f32, 1usize), (c, 0.92, 2), (d, 0.5, 3)] {
        let v = mix(&vecs[0].1, w, &vecs[own].1, (1.0 - w * w).sqrt());
        s.vera.lane.store.insert(key_of(&s, id), &v);
    }
    let r = s.execute("photo.duplicates", &json!({"id": a.0})).unwrap();
    let list = rows(&r);
    let similar: Vec<&Value> = list.iter().filter(|h| h["tier"] == "similar").collect();
    assert_eq!(similar.len(), 2, "{r}");
    assert_eq!(similar[0]["id"], b.0, "the closest first: {r}");
    assert_eq!(similar[0]["lane"], "vector");
    assert_eq!(similar[0]["strong"], true, "{r}");
    assert_eq!(similar[0]["pdq"], 256, "the fingerprints say different: {r}");
    assert_eq!(similar[1]["id"], c.0);
    assert_eq!(similar[1]["strong"], false);
    assert!(similar[1]["pdq"].is_null(), "no fingerprint for c: {r}");
    assert!(similar.iter().all(|h| h["cosine"].as_f64().unwrap() >= f64::from(lightcraft_vera::thresholds::EMBED_NEAR)));
    assert!(!list.iter().any(|h| h["id"] == d.0), "{r}");
    assert_eq!(r["embedded"], true);
    // a digest match stays in its own tier, ahead of the similar ones
    s.vera.digests.insert(key_of(&s, c), far_a);
    let r = s.execute("photo.duplicates", &json!({"id": a.0})).unwrap();
    assert_eq!(rows(&r)[0]["id"], c.0, "{r}");
    assert_eq!(rows(&r)[0]["tier"], "picture");
    assert!(rows(&r)[0].get("cosine").is_none());
    // ignored pairs leave the similar tier too
    s.execute("photo.ignorePair", &json!({"a": a.0, "b": b.0})).unwrap();
    let r = s.execute("photo.duplicates", &json!({"id": a.0})).unwrap();
    assert!(!rows(&r).iter().any(|h| h["id"] == b.0), "{r}");
    assert_eq!(r["ignored"], 1);
}

#[test]
fn vera_config_is_saved_with_the_library() {
    let dir = temp_dir("config");
    let lib = dir.join("lib");
    let mut s = Session::new();
    s.open_library(&lib, true).unwrap();
    let r = s.execute("vera.config", &json!({})).unwrap();
    assert_eq!(r["vectorLane"], true, "on by default: {r}");
    assert!(s.execute("vera.config", &json!({"vectorLane": "off"})).is_err());
    let r = s.execute("vera.config", &json!({"vectorLane": false})).unwrap();
    assert_eq!(r["vectorLane"], false);
    let prefs: Value = serde_json::from_slice(&std::fs::read(lib.join("prefs.json")).unwrap()).unwrap();
    assert_eq!(prefs["vera"], json!({"vectorLane": false}), "{prefs}");
    s.close_library().unwrap();
    drop(s);
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    assert_eq!(s.execute("vera.config", &json!({})).unwrap()["vectorLane"], false, "read back");
    s.execute("vera.config", &json!({"vectorLane": true})).unwrap();
    s.close_library().unwrap();
    drop(s);
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    assert_eq!(s.execute("vera.config", &json!({})).unwrap()["vectorLane"], true);
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vera_embeddings_follow_the_library_folder() {
    let dir = temp_dir("sidecar");
    let lib = dir.join("lib");
    let mut s = Session::new();
    s.open_library(&lib, true).unwrap();
    s.execute("vera.status", &json!({})).unwrap();
    let ids = distinct(&s);
    for (i, id) in ids.iter().take(3).enumerate() {
        let key = key_of(&s, *id);
        assert!(s.vera.lane.store.insert(key, &random_vec(i as u64)));
    }
    s.close_library().unwrap();
    let file = lib.join(format!("vera-embeddings-{}.bin", super::lane::MODEL_ID));
    assert!(file.is_file());
    drop(s);
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(st["embedded"], 3, "{st}");
    assert!(st["sidecars"][1]["bytes"].as_u64().is_some_and(|n| n > 3 * DIM as u64 * 4), "{st}");
    // removing the model deletes the model files and this library's embeddings
    let models = dir.join("models").join(super::lane::MODEL_ID);
    fake_model(&models);
    std::fs::write(dir.join("models").join("keep-me.txt"), b"other files are left alone").unwrap();
    s.vera_model_dir = Some(models.clone());
    let r = s.execute("vera.model.remove", &json!({})).unwrap();
    assert_eq!((r["removed"].as_bool(), r["files"].as_u64(), r["sidecar"].as_bool()), (Some(true), Some(3), Some(true)), "{r}");
    assert!(!file.exists() && !models.exists());
    assert!(dir.join("models").join("keep-me.txt").is_file());
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["embedded"].as_u64(), st["model"]["installed"].as_bool()), (Some(0), Some(false)), "{st}");
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn vera_model_download_needs_the_licence_accepted() {
    let mut s = Session::with_demo();
    for p in [json!({}), json!({"acceptLicense": false}), json!({"acceptLicense": "yes"})] {
        let e = s.execute("vera.model.download", &p).unwrap_err().to_string();
        assert!(e.contains("Apache License 2.0") && e.contains("acceptLicense"), "{e}");
    }
    // accepted, but this session has nowhere to keep a model
    let e = s.execute("vera.model.download", &json!({"acceptLicense": true})).unwrap_err().to_string();
    assert!(e.contains("nowhere"), "{e}");
    // already installed: nothing to fetch (no network touched)
    let dir = temp_dir("download");
    fake_model(&dir);
    s.vera_model_dir = Some(dir.clone());
    let r = s.execute("vera.model.download", &json!({"acceptLicense": true})).unwrap();
    assert_eq!((r["started"].as_bool(), r["installed"].as_bool()), (Some(false), Some(true)), "{r}");
    assert_eq!(s.execute("vera.model.cancel", &json!({})).unwrap()["cancelled"], false);
    let st = s.execute("vera.model.status", &json!({})).unwrap();
    assert_eq!(st["installed"], true);
    assert_eq!(st["modelId"], super::lane::MODEL_ID);
    assert_eq!(st["licence"]["name"], "Apache License 2.0");
    assert!(st["licence"]["notice"].as_str().unwrap().contains("Google"));
    assert!(st["bytes"].as_u64().is_some_and(|b| b > 1_500_000_000), "{st}");
    assert!(st["dir"].as_str().is_some() && st.get("downloading").is_none(), "{st}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The whole lane on the real checkpoint: `LIGHTCRAFT_SIGLIP2_DIR=<dir> cargo test -p lightcraft-engine
/// vera_real_pump -- --ignored --nocapture`.
#[test]
#[ignore = "needs the SigLIP 2 checkpoint in LIGHTCRAFT_SIGLIP2_DIR"]
fn vera_real_pump_embeds_the_demo_library_and_finds_the_lake() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
        return;
    };
    let mut s = Session::with_demo();
    s.vera_model_dir = Some(dir);
    let photos = s.catalog.photos().filter(|p| super::eligible(p)).count();
    let started = std::time::Instant::now();
    let mut last = Value::Null;
    let mut saw_row = false;
    while started.elapsed().as_secs() < 1200 {
        last = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
        saw_row |= s.activity.list().iter().any(|t| t.kind == "vera" && t.label == "Embedding photos");
        let e = &last["embedding"];
        assert_ne!(e["state"], "error", "{last}");
        if e["pending"] == 0 && e["embedded"].as_u64() == Some(photos as u64) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let took = started.elapsed().as_secs_f64();
    eprintln!("embedded {photos} photos in {took:.1} s ({:.2} photos/s): {last}", photos as f64 / took);
    assert!(saw_row, "the activity stack showed the work");
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["embedded"].as_u64(), st["pendingEmbeddings"].as_u64()), (Some(photos as u64), Some(0)), "{st}");
    assert_eq!(st["lanes"]["vector"]["state"], "live");
    assert!(!s.activity.list().iter().any(|t| t.label == "Embedding photos"), "the row is gone once the work is done");

    // the raw numbers behind the cutoff: the space's background (image pairs) and this query's cosines
    eprintln!("space background (image pairs, used by photo.similar): {:?}", s.vera.lane.store.background());
    let title_of_key = |s: &Session, k: u64| -> String {
        let key = s.vera.lane.store.key_of(k).unwrap().to_string();
        s.catalog.photos().find(|p| content_key(p) == key).map(|p| p.meta.title.clone()).unwrap_or_default()
    };
    let raw = |s: &mut Session, q: &str, n: usize| -> Vec<(String, f32)> {
        let qv = super::lane::query_vector(s, q).unwrap().unwrap();
        let near = s.vera.lane.store.space().unwrap().nearest(&qv, n, None);
        near.iter().map(|(k, c)| (title_of_key(s, *k), (c * 1e4).round() / 1e4)).collect()
    };
    let t = std::time::Instant::now();
    let r = s.execute("search.text", &json!({"query": "lake at dawn", "filter": false})).unwrap();
    eprintln!("search.text 'lake at dawn' in {:.0} ms (the text tower loads on the first query): {r}", t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(r["lanes"], json!(["keyword", "vector"]), "{r}");
    let lakes = ["Lake at first light", "Still water", "Alpenglow"];
    let all = raw(&mut s, "lake at dawn", 100);
    eprintln!("'lake at dawn' cosines: {all:?}");
    assert!(all.iter().take(3).all(|(t, _)| lakes.contains(&t.as_str())), "the model ranks the three lake scenes first: {all:?}");
    let first = &rows(&r)[0];
    let lake = s.catalog.photos().find(|p| p.meta.title == "Lake at first light").map(|p| p.id.0);
    assert_eq!(first["id"].as_u64(), lake, "the lake scene first: {r}");
    eprintln!(
        "'lake at dawn': first row lane {} (vector rows {} above the query's noise ceiling {})",
        first["lane"],
        rows(&r).iter().filter(|h| h["lane"] != "keyword").count(),
        r["cutoff"]
    );
    let t = std::time::Instant::now();
    let again = s.execute("search.text", &json!({"query": "lake at dawn", "filter": false})).unwrap();
    eprintln!("the same query again (cached vector) in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(again["photos"], r["photos"]);
    // queries the model is sure of keep their scenes above the ceiling, first, found by both lanes or by meaning
    for (q, want) in [
        ("the northern lights", &["Aurora over the ridge", "Green curtains"][..]),
        ("sand dunes in the desert", &["Dune ridges", "Wind lines", "Afternoon sand"][..]),
        ("a pink flower close up", &["Violet bloom", "Cosmos bloom", "Pink petals"][..]),
        ("a field of lavender", &["Lavender rows", "Evening fields"][..]),
    ] {
        let r = s.execute("search.text", &json!({"query": q, "filter": false})).unwrap();
        let shown: Vec<(String, Value)> = rows(&r)
            .iter()
            .take(4)
            .map(|h| (s.catalog.photo(PhotoId(h["id"].as_u64().unwrap())).unwrap().meta.title.clone(), h["lane"].clone()))
            .collect();
        eprintln!("{q:?}: cutoff {}, raw top {:?}, shown {shown:?}", r["cutoff"], raw(&mut s, q, 3));
        assert!(want.contains(&shown[0].0.as_str()), "{q}: {shown:?}");
        assert!(shown[0].1 == "vector" || shown[0].1 == "both", "{q}: {shown:?}");
    }
    // a query about nothing in the library keeps no vector rows: the ceiling is above all of them
    for q in ["a red sports car", "a dog"] {
        let r = s.execute("search.text", &json!({"query": q, "filter": false})).unwrap();
        eprintln!("{q:?}: cutoff {}, raw top {:?}", r["cutoff"], raw(&mut s, q, 3));
        assert!(rows(&r).iter().all(|h| h["lane"] == "keyword"), "{q}: {r}");
    }

    let lake = PhotoId(lake.unwrap());
    let sim = s.execute("photo.similar", &json!({"id": lake.0, "filter": false})).unwrap();
    eprintln!("photo.similar on the lake: {sim}");
    let key = key_of(&s, lake);
    let v = s.vera.lane.store.get(&key).unwrap().to_vec();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-4, "unit norm: {n}");
    let space_bg = s.vera.lane.store.background().unwrap();
    let space = s.vera.lane.store.space().unwrap();
    let near = space.nearest(&v, 100, s.vera.lane.store.id_of(&key));
    let named: Vec<(String, f32)> = near.iter().map(|(k, c)| (title_of_key(&s, *k), (c * 1e4).round() / 1e4)).collect();
    let qbg = super::commands::query_background(space, &v).unwrap();
    let others = space.len() - 1;
    let (space_ceiling, query_ceiling) = (lightcraft_vera::cutoff::ceiling(&space_bg, others), lightcraft_vera::cutoff::ceiling(&qbg, others));
    eprintln!(
        "lake image cosines: {named:?}\n  space-background ceiling {space_ceiling:.4} ({space_bg:?}); per-query ceiling {query_ceiling:.4} ({qbg:?})"
    );
    // the ranking photo.similar cuts: unit vectors, best first, the other lake scene at the top
    assert!(near.windows(2).all(|w| w[0].1 >= w[1].1), "rank order: {named:?}");
    assert!(near.iter().all(|(_, c)| (-1.0..=1.0001).contains(c)), "{named:?}");
    assert_eq!(named[0].0, "Still water", "{named:?}");
    let space_n: Vec<f32> = space.iter().map(|(_, v)| v.iter().map(|x| x * x).sum::<f32>().sqrt()).collect();
    assert!(space_n.iter().all(|n| (n - 1.0).abs() < 1e-4), "every stored vector is a unit vector");
    // what it returns: the rows above the space's ceiling, in that order, with the cosines of the stored vectors
    let cut = sim["cutoff"].as_f64().unwrap();
    assert!((cut - f64::from(space_ceiling)).abs() < 1e-3, "{sim}");
    let cos: Vec<f64> = rows(&sim).iter().map(|h| h["cosine"].as_f64().unwrap()).collect();
    assert!(cos.windows(2).all(|w| w[0] >= w[1]) && cos.iter().all(|c| *c >= cut - 1e-4), "{sim}");
    assert_eq!(cos.len(), near.iter().filter(|(_, c)| f64::from(*c) >= cut).count(), "{sim}");
    let dups = s.execute("photo.duplicates", &json!({"id": lake.0})).unwrap();
    eprintln!("photo.duplicates on the lake: {dups}");
}

/// Gate 2's JPEG re-exports (the embed crate's tests have no encoder): the fixture scene at 1024, display-encoded,
/// against itself written as a JPEG at quality 85 and 50 (4:2:0) and read back, all through the worker's squash.
/// `LIGHTCRAFT_SIGLIP2_DIR=<dir> cargo test -p lightcraft-engine vera_gate2 -- --ignored --nocapture`.
#[test]
#[ignore = "needs the SigLIP 2 checkpoint in LIGHTCRAFT_SIGLIP2_DIR"]
fn vera_gate2_jpeg_re_exports() {
    use lightcraft_codecs::{ChromaSubsampling, DecodeOptions, EncodeImage, EncodeMeta};
    use lightcraft_raster::Rgba8;
    let Some(dir) = std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
        return;
    };
    let srgb = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    };
    let to8 = |v: f32| (srgb(v) * 255.0).round() as u8;
    let scene = &lightcraft_scenes::demo_library()[0];
    let (w, h) = (1024, (1024.0 / scene.aspect()).round() as usize);
    let lin = lightcraft_scenes::render_srgb_linear(scene.kind, scene.seed, w, h);
    let original = Rgba8 { width: w, height: h, data: lin.data.iter().map(|p| [to8(p[0]), to8(p[1]), to8(p[2]), 255]).collect() };
    let m = lightcraft_embed::Model::load(&dir, lightcraft_embed::best_device()).unwrap();
    let embed = |img: &Rgba8| m.embed_image(&super::worker::squash(img, lightcraft_embed::INPUT).unwrap()).unwrap();
    let base = embed(&original);
    for q in [85u8, 50] {
        let jpeg = lightcraft_codecs::encode_jpeg(&EncodeImage::rgba8(&original), q, ChromaSubsampling::S420, &EncodeMeta::default()).unwrap();
        let back = lightcraft_codecs::decode(&jpeg, DecodeOptions::default()).unwrap().image;
        let img = Rgba8 { width: back.width, height: back.height, data: back.data.iter().map(|p| [to8(p[0]), to8(p[1]), to8(p[2]), 255]).collect() };
        let c = lightcraft_vera::cutoff::cosine(&base, &embed(&img));
        eprintln!("Gate 2: 1024 frame vs its JPEG q{q} 4:2:0 ({} bytes): cosine {c:.4}", jpeg.len());
        assert!(c > 0.9, "q{q}: {c}");
    }
}
