//! Agreement with the upstream Python conformance validator (the oracle).
//!
//! Runs only when `VER_ORACLE_PY` (a Python interpreter) and `VER_ORACLE_SRC` (the
//! `packages/ver-validator/src` directory) are set; `cargo xtask ver oracle` sets them. Without
//! them each test prints that it was skipped and passes.

use crate::common::*;
use lightcraft_ver::record::{Level, Schema};
use lightcraft_ver::validate;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

fn oracle() -> Option<(String, String)> {
    match (std::env::var("VER_ORACLE_PY"), std::env::var("VER_ORACLE_SRC")) {
        (Ok(py), Ok(src)) if !py.is_empty() && !src.is_empty() => Some((py, src)),
        _ => None,
    }
}

/// The oracle's JSON report for one file: (status, issue codes).
fn run(py: &str, src: &str, file: &Path, args: &[String]) -> (String, Vec<(String, String)>) {
    let out = Command::new(py)
        .args(["-m", "ver_validator.cli", "--format", "json"])
        .args(args)
        .arg(file)
        .env("PYTHONPATH", src)
        .output()
        .expect("run the oracle");
    let report: Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("oracle output for {}: {e}: {}", file.display(), String::from_utf8_lossy(&out.stderr)));
    let r = &report["results"][0]["report"];
    let codes = r["issues"]
        .as_array()
        .map(|v| {
            v.iter().map(|i| (i["code"].as_str().unwrap_or_default().to_string(), i["severity"].as_str().unwrap_or_default().to_string())).collect()
        })
        .unwrap_or_default();
    (report["status"].as_str().unwrap_or_default().to_string(), codes)
}

#[test]
fn oracle_finds_built_records_conformant() {
    let Some((py, src)) = oracle() else {
        eprintln!("skipped: VER_ORACLE_PY / VER_ORACLE_SRC not set");
        return;
    };
    let dir = std::env::temp_dir().join(format!("lightcraft-ver-oracle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    for name in ["jpeg-orientation-6.jpg", "png-srgb-icc.png", "gif-frame0-subrectangle.gif"] {
        for (schema, flag) in [(Schema::V1_0_1, "1.0.1"), (Schema::V1_1Draft, "1.1-draft")] {
            let rec = l3_record(&corpus_file(name), schema).expect("record");
            assert_eq!(rec.0["conformance_level"].as_str(), Some(Level::L3.as_str()));
            let path = dir.join(format!("{name}-{flag}.json"));
            std::fs::write(&path, rec.to_json_pretty()).expect("write");
            let (status, codes) = run(&py, &src, &path, &["--schema".into(), flag.into()]);
            let errors: Vec<_> = codes.iter().filter(|(_, s)| s == "error").collect();
            assert_eq!(status, "conformant", "{name} {flag}: {codes:?}");
            assert!(errors.is_empty(), "{name} {flag}: {errors:?}");
            assert!(codes.iter().all(|(c, _)| ["VER308", "VER404", "VER1206"].contains(&c.as_str())), "{name} {flag}: {codes:?}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn oracle_agrees_on_every_claimed_code() {
    let Some((py, src)) = oracle() else {
        eprintln!("skipped: VER_ORACLE_PY / VER_ORACLE_SRC not set");
        return;
    };
    let root = standard().join("conformance");
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).expect("manifest")).expect("json");
    let claimed: BTreeSet<&str> = validate::PROFILE_CODES.iter().chain(validate::STRUCTURAL_CODES).copied().collect();
    let mut disagreements = Vec::new();
    let mut compared = 0;
    let mut agreed_codes = 0;
    for kind in ["valid", "invalid"] {
        for e in manifest[kind].as_array().expect("entries") {
            let file = e["file"].as_str().expect("file");
            let path = root.join(file);
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(doc) = serde_json::from_str::<Value>(&text) else { continue };
            let schema_flag = e.get("schema").and_then(Value::as_str).unwrap_or("1.0.1");
            let schema = if schema_flag == "1.1-draft" { Schema::V1_1Draft } else { Schema::V1_0_1 };
            let mut args = vec!["--schema".to_string(), schema_flag.to_string()];
            let level = e.get("level").and_then(Value::as_str);
            if let Some(l) = level {
                args.extend(["--level".to_string(), l.to_string()]);
            }
            for (key, flag) in [("spaces_registry", "--spaces"), ("signature_key", "--signature-key"), ("registry", "--registry")] {
                if let Some(p) = e.get(key).and_then(Value::as_str) {
                    args.extend([flag.to_string(), root.join(p).display().to_string()]);
                }
            }
            for p in e.get("lineage_peers").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                args.extend(["--lineage-peer".to_string(), root.join(p).display().to_string()]);
            }
            let (_, theirs) = run(&py, &src, &path, &args);
            let level = level.and_then(|l| match l {
                "L0" => Some(Level::L0),
                "L1" => Some(Level::L1),
                "L2" => Some(Level::L2),
                "L3" => Some(Level::L3),
                _ => None,
            });
            let mut ours = validate::structural(&doc, schema);
            ours.extend(validate::profile_at(&doc, level));
            let theirs: BTreeSet<&str> = theirs.iter().map(|(c, _)| c.as_str()).filter(|c| claimed.contains(c)).collect();
            let ours: BTreeSet<&str> = ours.iter().map(|i| i.code.as_str()).filter(|c| claimed.contains(c)).collect();
            agreed_codes += theirs.intersection(&ours).count();
            if theirs != ours {
                disagreements.push(format!("{file}: oracle {theirs:?}, ours {ours:?}"));
            }
            compared += 1;
        }
    }
    eprintln!("compared {compared} fixtures with the oracle; {agreed_codes} claimed codes reported by both");
    assert!(compared >= 90 && agreed_codes >= 60, "{compared} fixtures, {agreed_codes} codes");
    assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
}
