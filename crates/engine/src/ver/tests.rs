//! VER Records end to end: a demo JPEG's Record at L3 under both schemas (checked by the Python oracle when it is
//! installed), the raw and no-model refusals, `ver.export` → `ver.import` round trips, tampering, the export hook and
//! the digest lane's Canonical-Buffer fingerprints. No model is needed: the tests embed with [`TestSource`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use super::source::{TEST_SPACE, TestSource};
use crate::Session;
use crate::export::{ExportOptions, export_photo};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lc-ver-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `n` demo scenes exported as JPEGs (with their XMP: title, keywords, creator) into `dir`.
fn demo_jpegs(dir: &Path, n: usize, long_edge: u32) -> Vec<PathBuf> {
    let mut s = Session::with_demo();
    let ids: Vec<PhotoId> = s.catalog.photos().filter(|p| crate::vera::eligible(p) && p.copy_of.is_none()).map(|p| p.id).take(n).collect();
    let o = ExportOptions::from_json(&json!({"format": "jpeg", "longEdge": long_edge, "quality": 90}));
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let e = export_photo(&mut s, *id, &o, i + 1).unwrap();
            let p = dir.join(format!("demo-{i}.jpg"));
            std::fs::write(&p, &e.bytes).unwrap();
            p
        })
        .collect()
}

/// A library at `lib` with `files` imported, embedding with the test source.
fn library(lib: &Path, files: &[PathBuf]) -> (Session, Vec<PhotoId>) {
    let mut s = Session::new().with_fs();
    s.open_library(lib, false).unwrap();
    s.ver.test_source = Some(Arc::new(TestSource));
    let paths: Vec<String> = files.iter().map(|f| f.to_string_lossy().to_string()).collect();
    let r = s.execute("library.import", &json!({"paths": paths})).unwrap();
    let ids: Vec<PhotoId> = r["imported"].as_array().unwrap().iter().map(|v| PhotoId(v.as_u64().unwrap())).collect();
    assert_eq!(ids.len(), files.len(), "{r}");
    (s, ids)
}

fn codes(issues: &Value, severity: &str) -> Vec<String> {
    issues
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["severity"].as_str().is_some_and(|s| s.eq_ignore_ascii_case(severity)))
        .map(|i| i["code"].as_str().unwrap().to_string())
        .collect()
}

/// The upstream Python validator, when it is installed (`VER_ORACLE_PY` + `VER_ORACLE_SRC`, or the paths this work
/// was done with): its JSON report of `record`, else `None`.
fn oracle(record: &Value, schema: &str, key: Option<&Path>) -> Option<Value> {
    const PY: &str = "/tmp/claude-0/-home-user-rlp-main/a4587cfa-3d8f-5403-9e32-64a234ca7930/scratchpad/ver-venv/bin/python";
    const SRC: &str =
        "/tmp/claude-0/-home-user-rlp-main/a4587cfa-3d8f-5403-9e32-64a234ca7930/scratchpad/vera-upstream/vera-main/packages/ver-validator/src";
    let py = std::env::var("VER_ORACLE_PY").unwrap_or_else(|_| PY.into());
    let src = std::env::var("VER_ORACLE_SRC").unwrap_or_else(|_| SRC.into());
    if !Path::new(&py).exists() || !Path::new(&src).exists() {
        eprintln!("ver: the Python oracle is not installed; skipped");
        return None;
    }
    let file = std::env::temp_dir().join(format!("lc-ver-oracle-{}-{}.json", std::process::id(), super::util::uuid4("oracle")));
    std::fs::write(&file, serde_json::to_vec_pretty(record).unwrap()).unwrap();
    let mut cmd = std::process::Command::new(py);
    cmd.env("PYTHONPATH", src).args(["-m", "ver_validator.cli", "--schema", schema, "--format", "json"]);
    if let Some(k) = key {
        cmd.arg("--signature-key").arg(k);
    }
    let out = cmd.arg(&file).output().ok()?;
    let _ = std::fs::remove_file(&file);
    let report: Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("oracle output: {e}: {}", String::from_utf8_lossy(&out.stdout)));
    Some(report["results"][0]["report"].clone())
}

/// The oracle's verdict on `record`: conformant, and only the warnings LightCraft's Records carry by design.
fn assert_oracle_clean(record: &Value, schema: &str, key: Option<&Path>, allowed: &[&str]) {
    let Some(r) = oracle(record, schema, key) else { return };
    assert_eq!(r["status"], "conformant", "oracle on {schema}: {r:#}");
    let warnings: Vec<&str> = r["issues"].as_array().unwrap().iter().map(|i| i["code"].as_str().unwrap()).collect();
    assert!(warnings.iter().all(|c| allowed.contains(c)), "oracle warnings {warnings:?} (allowed {allowed:?}): {r:#}");
}

#[test]
fn an_l3_record_of_a_demo_jpeg_validates_under_both_schemas() {
    let dir = temp_dir("l3");
    let files = demo_jpegs(&dir, 1, 640);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    let r = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap();
    assert_eq!(r["conformant"], true, "{r:#}");
    assert_eq!(r["level"], "L3", "{r:#}");
    assert_eq!(r["schema"], "1.1-draft");
    assert!(codes(&r["issues"], "error").is_empty(), "{}", r["issues"]);
    let rec = &r["record"];
    assert_eq!(rec["ver_version"], "1.1");
    assert_eq!(rec["cpnp_version"], "1.0");
    assert_eq!(rec["conformance_level"], "L3");
    assert_eq!(rec["image"]["format"], "jpeg");
    let content = rec["identity"]["content_hash"]["value"].as_str().unwrap().to_string();
    assert_eq!(content, super::util::sha256_hex(&std::fs::read(&files[0]).unwrap()));
    assert_eq!(rec["image"]["source_bytes_ref"]["sha256"], content.as_str(), "E25");
    assert!(rec["image"]["source_bytes_ref"]["uri"].as_str().unwrap().starts_with("file:///"));
    // L1: every family stated, the XMP carried byte-exact
    for f in ["exif", "xmp", "iptc", "icc", "c2pa"] {
        assert!(rec["metadata"]["availability"][f].is_string(), "{f}");
    }
    assert_eq!(rec["metadata"]["availability"]["xmp"], "present");
    assert!(rec["metadata"]["raw"].as_array().unwrap().iter().any(|e| e["segment"] == "xmp"));
    // L2: the demo's title and keywords, from the file's XMP, with a trust tier each
    assert!(rec["metadata"]["normalized"]["dc:title"].is_string(), "{}", rec["metadata"]);
    let trust = rec["metadata"]["trust"].as_array().unwrap();
    assert!(trust.iter().all(|t| t["tier"] == "unsigned"));
    assert_eq!(trust.len(), rec["metadata"]["normalized"].as_object().unwrap().keys().filter(|k| !k.starts_with('@')).count());
    // L3: acquisition, registered actions in order, an honest C2PA status
    assert_eq!(rec["provenance"]["acquisition"]["method"], "user_upload");
    let actions: Vec<&str> = rec["provenance"]["chain"].as_array().unwrap().iter().map(|e| e["action"].as_str().unwrap()).collect();
    assert_eq!(actions, ["import", "decode", "normalize", "embed"]);
    assert_eq!(rec["provenance"]["c2pa"]["status"], "missing");
    // the embedding, in the test space, fp16, with its compute block (1.1 §9)
    assert_eq!(rec["embeddings"][0]["role"], "canonical_visual");
    assert_eq!(rec["embeddings"][0]["space_id"], TEST_SPACE);
    assert_eq!(rec["embeddings"][0]["dtype"], "fp16");
    assert!(rec["embeddings"][0]["vector_b64"].is_string());
    assert!(rec["embeddings"][0]["compute"].is_object());
    assert!(rec["spaces"][0].get("reference").is_none());
    assert!(rec["extensions"]["org.verstandard.lightcraft"]["field_origins"].is_object());
    assert_oracle_clean(rec, "1.1-draft", None, &["VER308", "VER404"]);

    // the 1.0 twin: the same bytes, another document
    let r10 = s.execute("ver.record", &json!({"id": ids[0].0, "schema": "1.0.1"})).unwrap();
    let rec10 = &r10["record"];
    assert_eq!(r10["conformant"], true, "{r10:#}");
    assert_eq!(rec10["ver_version"], "1.0");
    assert_eq!(rec10["identity"]["content_hash"]["value"], content.as_str());
    assert_eq!(rec10["identity"]["pixel_hash"], rec["identity"]["pixel_hash"]);
    assert_ne!(rec10["record_id"], rec["record_id"]);
    assert!(rec10.get("extensions").is_none() && rec10.get("lineage").is_none());
    assert_oracle_clean(rec10, "1.0.1", None, &["VER308", "VER404", "VER606"]);
    // `VER_DUMP_DIR=… cargo test …` keeps the two Records for a look
    if let Ok(d) = std::env::var("VER_DUMP_DIR") {
        std::fs::write(Path::new(&d).join("demo.ver.json"), serde_json::to_vec_pretty(rec).unwrap()).unwrap();
        std::fs::write(Path::new(&d).join("demo.ver-1.0.json"), serde_json::to_vec_pretty(rec10).unwrap()).unwrap();
    }
    // idempotent re-issue: the same record id for the same bytes (E16)
    let again = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap();
    assert_eq!(again["record"]["record_id"], rec["record_id"]);
    // a lower level asked for is a lower claim
    let l1 = s.execute("ver.record", &json!({"id": ids[0].0, "level": "L1"})).unwrap();
    assert_eq!(l1["record"]["conformance_level"], "L1");
    assert!(s.execute("ver.record", &json!({"id": ids[0].0, "level": "L9"})).is_err());
    assert!(s.execute("ver.record", &json!({"id": ids[0].0, "schema": "2.0"})).is_err());
    let st = s.execute("ver.status", &json!({})).unwrap();
    assert_eq!(st["ready"], true, "{st}");
    assert_eq!(st["lastProduced"]["level"], "L1", "{st}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_raw_has_no_record_and_no_model_means_none() {
    let dir = temp_dir("raw");
    let raw = dir.join("Shot.dng");
    std::fs::write(&raw, crate::tests_xmp::synthetic_dng_with(None, Default::default())).unwrap();
    let jpeg = demo_jpegs(&dir, 1, 320).remove(0);
    let (mut s, ids) = library(&dir.join("lib"), &[raw, jpeg]);
    let e = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap_err().to_string();
    assert!(e.contains("no_cpnp_path") && e.contains("raw"), "{e}");
    // without an embedding source (no model installed) there is no Record either
    s.ver.test_source = None;
    let e = s.execute("ver.record", &json!({"id": ids[1].0})).unwrap_err().to_string();
    assert!(e.contains("needs_model"), "{e}");
    let st = s.execute("ver.status", &json!({})).unwrap();
    assert_eq!(st["ready"], false);
    assert_eq!(st["model"]["code"], "needs_model", "{st}");
    // a demo photo has no file at all
    let mut demo = Session::with_demo();
    demo.ver.test_source = Some(Arc::new(TestSource));
    let id = demo.active().unwrap();
    let e = demo.execute("ver.record", &json!({"id": id.0})).unwrap_err().to_string();
    assert!(e.contains("no_asset"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ver_export_round_trips_through_import_and_a_reexport_matches_perceptually() {
    let dir = temp_dir("roundtrip");
    let files = demo_jpegs(&dir, 2, 640);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    let out = dir.join("records");
    let r = s.execute("ver.export", &json!({"ids": [ids[0].0, ids[1].0], "dir": out.to_string_lossy()})).unwrap();
    assert_eq!(r["written"], 2, "{r}");
    let written: Vec<String> = r["files"].as_array().unwrap().iter().map(|f| f.as_str().unwrap().to_string()).collect();
    assert_eq!(written.len(), 4, "a 1.1 Record and its 1.0 twin each: {written:?}");
    let first11 = written.iter().find(|f| f.ends_with(".ver.json")).unwrap();
    let first10 = written.iter().find(|f| f.ends_with(".ver-1.0.json")).unwrap();
    // named by content hash
    let content = super::util::sha256_hex(&std::fs::read(&files[0]).unwrap());
    assert!(written.iter().any(|f| f.ends_with(&format!("{content}.ver.json"))), "{written:?}");
    for f in [first11, first10] {
        let c = s.execute("ver.import", &json!({"path": f})).unwrap();
        assert_eq!(c["status"], "accepted", "{c:#}");
        let me = c["recordSummary"]["contentHash"].as_str().unwrap().to_string();
        let photo = if me == content { ids[0].0 } else { ids[1].0 };
        let by: Vec<&str> = c["matches"].as_array().unwrap().iter().filter(|m| m["id"] == photo).map(|m| m["by"].as_str().unwrap()).collect();
        assert!(by.contains(&"content") && by.contains(&"pixel_hash"), "{c:#}");
        assert!(by.contains(&"vector"), "the test space is this session's own: {c:#}");
    }
    // a re-export (smaller, re-encoded) is the same picture perceptually, not the same pixels; its sidecar names the
    // source in its lineage
    let o = ExportOptions::from_json(&json!({"format": "jpeg", "longEdge": 400, "quality": 80}));
    let to = crate::export::Destination { dir: dir.join("out").to_string_lossy().to_string(), exact: None };
    let rows = crate::export::export_batch(&mut s, &[ids[0]], &o, &to, &mut crate::export::write_file, &|p| Path::new(p).exists()).unwrap();
    let sidecars: Vec<&str> = rows[0]["sidecars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let sc11 = sidecars.iter().find(|p| p.ends_with(".ver.json")).expect("a 1.1 sidecar");
    assert!(sidecars.iter().any(|p| p.ends_with(".ver-1.0.json")), "{sidecars:?}");
    let rec: Value = serde_json::from_slice(&std::fs::read(sc11).unwrap()).unwrap();
    assert_eq!(rec["lineage"]["parents"][0]["content_hash"], content.as_str(), "{:#}", rec["lineage"]);
    assert_eq!(rec["lineage"]["parents"][0]["relation"], "derived");
    let actions: Vec<&str> = rec["provenance"]["chain"].as_array().unwrap().iter().map(|e| e["action"].as_str().unwrap()).collect();
    assert_eq!(actions, ["resize", "export", "decode", "normalize", "embed"], "{actions:?}");
    assert_eq!(rec["identity"]["content_hash"]["value"], super::util::sha256_hex(&std::fs::read(rows[0]["path"].as_str().unwrap()).unwrap()));
    assert_oracle_clean(&rec, "1.1-draft", None, &["VER308", "VER404"]);
    let c = s.execute("ver.import", &json!({"path": sc11})).unwrap();
    assert_eq!(c["status"], "accepted", "{c:#}");
    let mine: Vec<&Value> = c["matches"].as_array().unwrap().iter().filter(|m| m["id"] == ids[0].0).collect();
    assert!(mine.iter().any(|m| m["by"] == "pdq" || m["by"] == "phash"), "{c:#}");
    assert!(!mine.iter().any(|m| m["by"] == "pixel_hash" || m["by"] == "content"), "{c:#}");
    // the record given inline works the same, and bad input is refused
    assert_eq!(s.execute("ver.import", &json!({"record": rec})).unwrap()["status"], "accepted");
    assert!(s.execute("ver.import", &json!({})).is_err());
    assert!(s.execute("ver.import", &json!({"record": "not json"})).is_err());
    assert!(s.execute("ver.import", &json!({"record": {"ver_version": "9.9"}})).is_err());
    assert!(s.execute("ver.export", &json!({"ids": [ids[0].0]})).is_err(), "dir is required");
    assert!(s.execute("ver.export", &json!({"ids": [ids[0].0], "dir": out.to_string_lossy(), "schema": "2"})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_record_is_rejected() {
    let dir = temp_dir("tamper");
    let files = demo_jpegs(&dir, 1, 320);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    let rec = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap()["record"].clone();
    // the pixel hash's width no longer agrees with the image's (errata E2)
    let mut t = rec.clone();
    let w = t["identity"]["pixel_hash"]["width"].as_u64().unwrap();
    t["identity"]["pixel_hash"]["width"] = json!(w + 1);
    let c = s.execute("ver.import", &json!({"record": t})).unwrap();
    assert_eq!(c["status"], "rejected", "{c:#}");
    assert!(c["matches"].as_array().unwrap().is_empty());
    assert!(codes(&c["issues"], "error").contains(&"LCVER-E2".to_string()), "{c:#}");
    // a width no schema admits
    t["identity"]["pixel_hash"]["width"] = json!(0);
    let c = s.execute("ver.import", &json!({"record": t})).unwrap();
    assert_eq!(c["status"], "rejected");
    assert!(c["issues"].as_array().unwrap().iter().any(|i| i["pointer"].as_str().is_some_and(|p| p.starts_with("/identity/pixel_hash"))), "{c:#}");
    // an embedding whose bytes do not match its dim
    let mut t = rec.clone();
    t["embeddings"][0]["vector_b64"] = json!("AAAA");
    assert_eq!(s.execute("ver.import", &json!({"record": t})).unwrap()["status"], "rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_export_hook_writes_records_beside_a_jpeg_and_notes_why_not_for_a_raw_without_the_model() {
    let dir = temp_dir("hook");
    let raw = dir.join("Shot.dng");
    std::fs::write(&raw, crate::tests_xmp::synthetic_dng_with(None, Default::default())).unwrap();
    let jpeg = demo_jpegs(&dir, 1, 480).remove(0);
    let (mut s, ids) = library(&dir.join("lib"), &[jpeg, raw]);
    let to = crate::export::Destination { dir: dir.join("out").to_string_lossy().to_string(), exact: None };
    let o = ExportOptions::from_json(&json!({"format": "jpeg", "longEdge": 0}));
    let rows = crate::export::export_batch(&mut s, &[ids[0], ids[1]], &o, &to, &mut crate::export::write_file, &|p| Path::new(p).exists()).unwrap();
    for row in &rows {
        let sc: Vec<&str> = row["sidecars"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        assert!(sc.iter().any(|p| p.ends_with(".ver.json")) && sc.iter().any(|p| p.ends_with(".ver-1.0.json")), "{row}");
        assert!(row.get("notes").is_none(), "{row}");
    }
    // the raw's rendition names its source in the extensions: a raw has no Record to point at
    let raw_rec: Value = serde_json::from_slice(&std::fs::read(rows[1]["path"].as_str().unwrap().replace(".jpg", ".ver.json")).unwrap()).unwrap();
    assert!(raw_rec.get("lineage").is_none(), "{raw_rec:#}");
    let note = &raw_rec["extensions"]["org.verstandard.lightcraft"]["source"];
    assert_eq!(note["content_hash"], super::util::sha256_hex(&std::fs::read(dir.join("Shot.dng")).unwrap()));
    let actions: Vec<&str> = raw_rec["provenance"]["chain"].as_array().unwrap().iter().map(|e| e["action"].as_str().unwrap()).collect();
    assert_eq!(actions[0], "color_adjust", "a raw is always developed: {actions:?}");
    // the 1.0 twin carries only the actions 1.0 has
    let raw10: Value = serde_json::from_slice(&std::fs::read(rows[1]["path"].as_str().unwrap().replace(".jpg", ".ver-1.0.json")).unwrap()).unwrap();
    let a10: Vec<&str> = raw10["provenance"]["chain"].as_array().unwrap().iter().map(|e| e["action"].as_str().unwrap()).collect();
    assert_eq!(a10, ["decode", "normalize", "embed"]);
    assert_oracle_clean(&raw10, "1.0.1", None, &["VER308", "VER404", "VER606"]);
    // without the model: the export still happens, and its result says why there is no Record
    s.ver.test_source = None;
    let rows = crate::export::export_batch(&mut s, &[ids[1]], &o, &to, &mut crate::export::write_file, &|p| Path::new(p).exists()).unwrap();
    assert!(Path::new(rows[0]["path"].as_str().unwrap()).exists(), "{rows:?}");
    let notes = rows[0]["notes"].as_array().expect("a note");
    assert_eq!(notes.len(), 1);
    assert!(notes[0].as_str().unwrap().contains("model"), "{notes:?}");
    // switched off: no Records and no notes
    s.execute("ver.config", &json!({"exportSidecars": false})).unwrap();
    let rows = crate::export::export_batch(&mut s, &[ids[0]], &o, &to, &mut crate::export::write_file, &|p| Path::new(p).exists()).unwrap();
    assert!(rows[0]["sidecars"].as_array().unwrap().is_empty() && rows[0].get("notes").is_none(), "{rows:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Pump `vera.pump` until it is done (the fingerprinting thread does the work), at most `secs`.
fn pump_until_done(s: &mut Session, secs: u64) -> Value {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    loop {
        let r = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
        if r["active"] == false || std::time::Instant::now() > until {
            return r;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn the_digest_lane_hashes_pixels_for_jpegs_and_is_provisional_for_raws() {
    let dir = temp_dir("lane");
    let raw = dir.join("Shot.dng");
    std::fs::write(&raw, crate::tests_xmp::synthetic_dng_with(None, Default::default())).unwrap();
    let jpeg = demo_jpegs(&dir, 1, 640).remove(0);
    // the same pixels under other bytes: a comment segment after SOI changes the file, not the picture
    let mut twin_bytes = std::fs::read(&jpeg).unwrap();
    twin_bytes.splice(2..2, [0xFF, 0xFE, 0x00, 0x06, b'h', b'e', b'l', b'o']);
    let twin = dir.join("twin.jpg");
    std::fs::write(&twin, &twin_bytes).unwrap();
    let (mut s, ids) = library(&dir.join("lib"), &[jpeg, raw, twin]);
    let last = pump_until_done(&mut s, 60);
    assert_eq!(last["active"], false, "{last}");
    let key = |s: &Session, id: PhotoId| crate::media::content_key(s.catalog.photo(id).unwrap());
    let jf = *s.vera.digests.fingerprint(&key(&s, ids[0])).expect("the JPEG is fingerprinted");
    assert!(!jf.is_provisional() && jf.content.is_some(), "{jf:?}");
    let rf = *s.vera.digests.fingerprint(&key(&s, ids[1])).expect("the raw is fingerprinted");
    assert!(rf.is_provisional(), "{rf:?}");
    assert!(rf.content.is_some(), "the raw's bytes were read, so its content hash is known");
    let tf = *s.vera.digests.fingerprint(&key(&s, ids[2])).unwrap();
    assert_eq!(tf.pixel, jf.pixel, "one Canonical Buffer");
    assert_ne!(tf.content, jf.content);
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["pixelHashed"].as_u64(), st["provisional"].as_u64()), (Some(2), Some(1)), "{st}");
    // the same rendered pixels come first, as their own tier
    let d = s.execute("photo.duplicates", &json!({"id": ids[0].0})).unwrap();
    assert_eq!(d["photos"][0]["id"], ids[2].0, "{d}");
    assert_eq!(d["photos"][0]["tier"], "pixels", "{d}");
    // the fingerprints survive a reopen (format 2), with their hashes
    drop(s);
    let mut again = Session::new().with_fs();
    again.open_library(dir.join("lib"), false).unwrap();
    crate::vera::ensure_open(&mut again);
    assert_eq!(again.vera.digests.fingerprint(&key(&again, ids[0])), Some(&jf));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn signing_and_settings() {
    let dir = temp_dir("sign");
    let files = demo_jpegs(&dir, 1, 320);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    // settings are checked and saved with the library
    assert!(s.execute("ver.config", &json!({"authority": "Not A Domain"})).is_err());
    assert!(s.execute("ver.config", &json!({"authority": "verstandard"})).is_err());
    assert!(s.execute("ver.config", &json!({"exportSidecars": "yes"})).is_err());
    assert!(s.execute("ver.config", &json!({"defaultSchema": "2.0"})).is_err());
    assert!(s.execute("ver.config", &json!({"signKeyPath": dir.join("missing.pem").to_string_lossy()})).is_err());
    let r = s.execute("ver.config", &json!({"defaultSchema": "1.0.1"})).unwrap();
    assert_eq!(r["defaultSchema"], "1.0.1");
    assert_eq!(r["authority"], "org.verstandard");
    let rec = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap();
    assert_eq!(rec["record"]["ver_version"], "1.0");
    s.execute("ver.config", &json!({"defaultSchema": null})).unwrap();
    // an Ed25519 key, as `openssl genpkey -algorithm ed25519` writes it
    let signer = super::sign::Signer::from_seed(&[42; 32]);
    let mut der = vec![0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20];
    der.extend_from_slice(&[42; 32]);
    let key = dir.join("key.pem");
    std::fs::write(&key, format!("-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n", super::util::b64(&der))).unwrap();
    let public = dir.join("pub.pem");
    std::fs::write(&public, signer.public_pem()).unwrap();
    s.execute("ver.config", &json!({"signKeyPath": key.to_string_lossy()})).unwrap();
    let st = s.execute("ver.status", &json!({})).unwrap();
    assert_eq!(st["signing"]["keyId"], signer.key_id.as_str(), "{st}");
    let r = s.execute("ver.record", &json!({"id": ids[0].0})).unwrap();
    let rec = &r["record"];
    assert_eq!(rec["provenance"]["signature"]["format"], "jws-detached");
    assert_eq!(rec["provenance"]["signature"]["key_id"], signer.key_id.as_str());
    let actions: Vec<&str> = rec["provenance"]["chain"].as_array().unwrap().iter().map(|e| e["action"].as_str().unwrap()).collect();
    assert_eq!(actions.last(), Some(&"sign"));
    let mut pk = [0u8; 32];
    pk.copy_from_slice(&ed25519_dalek::SigningKey::from_bytes(&[42; 32]).verifying_key().to_bytes());
    assert!(super::sign::verify(rec, &pk).unwrap());
    // the oracle re-derives the payload (E17) and verifies it: no VER1402 / VER1403, and no VER308 now
    assert_oracle_clean(rec, "1.1-draft", Some(&public), &["VER404"]);
    // the settings persist in prefs.json
    drop(s);
    let mut again = Session::new().with_fs();
    again.open_library(dir.join("lib"), false).unwrap();
    let st = again.execute("ver.status", &json!({})).unwrap();
    assert_eq!(st["signing"]["enabled"], true, "{st}");
    assert!(st["producerId"].as_str().is_some_and(|p| p.starts_with("did:ver:local:")), "{st}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn background_ver_export_runs_at_the_pumps_pace() {
    let dir = temp_dir("bg");
    let files = demo_jpegs(&dir, 2, 320);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    let out = dir.join("records");
    let r = s
        .execute("ver.export", &json!({"ids": [ids[0].0, ids[1].0], "dir": out.to_string_lossy(), "schema": "1.1-draft", "background": true}))
        .unwrap();
    assert_eq!(r["queued"], 2, "{r}");
    // paused: nothing is written
    let p = s.execute("vera.pump", &json!({"pace": "pause"})).unwrap();
    assert_eq!(p["ver"]["pending"], 2, "{p}");
    assert!(s.activity.list().iter().any(|t| t.label == "Writing VER records"));
    let last = pump_until_done(&mut s, 60);
    assert_eq!(last["ver"]["written"], 2, "{last}");
    assert_eq!(std::fs::read_dir(&out).unwrap().count(), 2);
    assert!(!s.activity.list().iter().any(|t| t.label == "Writing VER records"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The per-photo cost of the Canonical-Buffer digest lane on the demo library: every demo scene written as a
/// full-size JPEG (quality 90), then fingerprinted canonically (full-size CPNP-1 + PDQ + pHash + SHA-256s) and, for
/// comparison, provisionally (the thumbnail-level decode the lane used before). Run with
/// `cargo test -p lightcraft-engine --release measure_canonical_lane_cost -- --ignored --nocapture`.
#[test]
#[ignore = "a measurement, not a check"]
fn measure_canonical_lane_cost() {
    let dir = temp_dir("measure");
    let mut s = Session::with_demo();
    let limit = std::env::var("VER_MEASURE_N").ok().and_then(|n| n.parse().ok()).unwrap_or(usize::MAX);
    let ids: Vec<PhotoId> = s.catalog.photos().filter(|p| crate::vera::eligible(p) && p.copy_of.is_none()).map(|p| p.id).take(limit).collect();
    let o = ExportOptions::from_json(&json!({"format": "jpeg", "longEdge": 0, "quality": 90}));
    s.ver.prefs.export_sidecars = Some(false);
    let mut files = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let e = export_photo(&mut s, *id, &o, i + 1).unwrap();
        let p = dir.join(format!("scene-{i}.jpg"));
        std::fs::write(&p, &e.bytes).unwrap();
        files.push((p, e.width, e.height));
    }
    let (mut canon, mut prov, mut px) = (0f64, 0f64, 0f64);
    for (p, w, h) in &files {
        let bytes = std::fs::read(p).unwrap();
        let t = std::time::Instant::now();
        let fp = crate::vera::fingerprint::canonical(&bytes).unwrap();
        let c = t.elapsed().as_secs_f64();
        assert!(!fp.is_provisional());
        let t = std::time::Instant::now();
        let (img, _) = crate::files::load_bytes(&bytes, 512).unwrap();
        crate::vera::fingerprint::digest_linear(&img).unwrap();
        let v = t.elapsed().as_secs_f64();
        eprintln!("{:>5}×{:<5} {:>7} bytes  canonical {:>7.1} ms  provisional {:>6.1} ms", w, h, bytes.len(), c * 1e3, v * 1e3);
        if std::env::var("VER_MEASURE_PARTS").is_ok() {
            use lightcraft_ver::hashing;
            let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let buf = lightcraft_ver::cpnp::canonicalize(&bytes).unwrap();
            let decode = ms(t);
            let t = std::time::Instant::now();
            hashing::content_hash(&bytes);
            let content = ms(t);
            let t = std::time::Instant::now();
            hashing::pixel_hash(&buf);
            let pixel = ms(t);
            let t = std::time::Instant::now();
            hashing::pdq(&buf);
            let pdq = ms(t);
            let t = std::time::Instant::now();
            hashing::phash_dct64(&buf);
            let phash = ms(t);
            let t = std::time::Instant::now();
            hashing::pdq_quality(&buf);
            let quality = ms(t);
            eprintln!(
                "    CPNP-1 {decode:.0} ms, content_hash {content:.0}, pixel_hash {pixel:.0}, PDQ {pdq:.0}, pHash {phash:.0}, quality {quality:.0}"
            );
        }
        canon += c;
        prov += v;
        px += (*w * *h) as f64;
    }
    let n = files.len() as f64;
    eprintln!(
        "{} scenes, mean {:.1} MP: canonical {:.1} ms/photo ({:.1} ms/MP), provisional {:.1} ms/photo",
        files.len(),
        px / n / 1e6,
        canon / n * 1e3,
        canon / (px / 1e6) * 1e3,
        prov / n * 1e3
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// LightCraft's own space descriptor (the one the model's Records carry) with a stand-in 768-number vector: what the
/// validators and the oracle say of it, without the model.
struct LightCraftSpace;

impl super::source::EmbeddingSource for LightCraftSpace {
    fn space(&self) -> Result<super::source::SpaceInfo, super::VerError> {
        Ok(super::source::SpaceInfo {
            space_id: super::space::space_id("org.verstandard"),
            dim: super::space::DIM,
            dtype: super::space::DTYPE.into(),
            descriptor: super::space::descriptor("org.verstandard", "7d241bb3becad218f211f480487f491df4f8c0a472ecf7afdec5615815a301f1"),
            producer: "lightcraft-embed/0.5.0 (candle 0.9.2, cpu, fp32)".into(),
            compute: json!({"runtime": "candle", "runtime_version": "0.9.2", "precision": "fp32", "device_class": "cpu"}),
        })
    }
    fn embed(&self, buf: &lightcraft_ver::cpnp::CanonicalBuffer) -> Result<Vec<f32>, super::VerError> {
        let input = super::space::model_input(&buf.rgb, buf.width as usize, buf.height as usize).unwrap();
        // a stand-in for the model: the mean of each of 256 blocks of the resized input, per channel
        let mut v = vec![0f32; super::space::DIM];
        for (i, p) in input.iter().enumerate() {
            let block = (i / super::space::INPUT / 16) * 16 + (i % super::space::INPUT) / 16;
            for c in 0..3 {
                v[block * 3 + c] += p[c];
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(v.iter().map(|x| x / n).collect())
    }
}

#[test]
fn the_lightcraft_space_descriptor_validates() {
    let dir = temp_dir("space");
    let files = demo_jpegs(&dir, 1, 480);
    let (mut s, ids) = library(&dir.join("lib"), &files);
    s.ver.test_source = Some(Arc::new(LightCraftSpace));
    for schema in ["1.1-draft", "1.0.1"] {
        let r = s.execute("ver.record", &json!({"id": ids[0].0, "schema": schema})).unwrap();
        assert_eq!(r["conformant"], true, "{r:#}");
        let rec = &r["record"];
        assert_eq!(rec["spaces"][0]["space_id"], "org.verstandard.space.siglip2-base-p32-256/1.0");
        assert_eq!(rec["spaces"][0]["model"]["checkpoint"], "google/siglip2-base-patch32-256");
        assert_eq!(rec["spaces"][0]["provenance"]["publisher"], "verstandard.org");
        // 768 fp16 components: 1536 bytes
        let bytes = super::util::unb64(rec["embeddings"][0]["vector_b64"].as_str().unwrap()).unwrap();
        assert_eq!(bytes.len(), 1536);
        let warnings = codes(&r["issues"], "warning");
        assert!(warnings.iter().all(|c| c == "VER308" || c == "VER404"), "{warnings:?}");
        assert_oracle_clean(rec, schema, None, &["VER308", "VER404"]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
