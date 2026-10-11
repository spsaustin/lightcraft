//! VER Records as commands (`docs/ver.md` → Commands): `ver.record`, `ver.export`, `ver.import`, `ver.status`,
//! `ver.config`. Every parameter is checked; none of them is journaled (a Record is derived, never part of the
//! catalog).

use std::path::PathBuf;

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd};
use crate::ver::{self, Level, Schema, VerError};
use crate::{EngineError, Result, Session};

/// Most photos one `ver.export` takes.
const MAX_IDS: usize = 10_000;

fn ver_err(c: &str, e: VerError) -> EngineError {
    bad(c, format!("{} ({})", e, e.code()))
}

fn schema_param(s: &Session, p: &Value, c: &str) -> Result<Schema> {
    match p.get("schema") {
        None | Some(Value::Null) => Ok(s.ver.prefs.default_schema()),
        Some(Value::String(v)) => ver::parse_schema(v).ok_or_else(|| bad(c, "`schema` must be \"1.1-draft\" or \"1.0.1\"")),
        Some(_) => Err(bad(c, "`schema` must be \"1.1-draft\" or \"1.0.1\"")),
    }
}

fn photo_of(s: &Session, p: &Value, c: &str) -> Result<PhotoId> {
    match p.get("id") {
        None | Some(Value::Null) => s.active().ok_or_else(|| bad(c, "no photo: give `id` or select one")),
        Some(v) => {
            let id = v.as_u64().map(PhotoId).ok_or_else(|| bad(c, "`id` must be a photo id"))?;
            s.catalog.photo(id).map(|_| id).ok_or_else(|| bad(c, format!("no photo {}", id.0)))
        }
    }
}

/// `ver.record {id?, schema?, level?}`.
fn record(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "ver.record";
    let id = photo_of(s, p, C)?;
    let schema = schema_param(s, p, C)?;
    let level = match p.get("level") {
        None | Some(Value::Null) => Level::L3,
        Some(v) => v.as_str().and_then(ver::parse_level).ok_or_else(|| bad(C, "`level` must be L0, L1, L2 or L3"))?,
    };
    let out = ver::produce(s, id, ver::Opts { schema, level }).map_err(|e| ver_err(C, e))?;
    let mut v = out.to_json();
    v["schema"] = json!(ver::schema_name(schema));
    Ok(v)
}

fn ids_param(s: &Session, p: &Value, c: &str) -> Result<Vec<PhotoId>> {
    match p.get("ids") {
        None | Some(Value::Null) => Ok(s.targets(&json!({}))),
        Some(Value::Array(a)) => {
            if a.len() > MAX_IDS {
                return Err(bad(c, format!("at most {MAX_IDS} photos at a time")));
            }
            a.iter().map(|v| v.as_u64().map(PhotoId).ok_or_else(|| bad(c, "`ids` must be photo ids"))).collect()
        }
        Some(_) => Err(bad(c, "`ids` must be a list of photo ids")),
    }
}

/// `ver.export {ids?, dir, schema?, background?}`.
fn export(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "ver.export";
    if cfg!(target_arch = "wasm32") {
        return Err(bad(C, "writing Records to a folder needs the desktop app"));
    }
    let dir = p.get("dir").and_then(Value::as_str).map(str::trim).filter(|d| !d.is_empty()).ok_or_else(|| bad(C, "missing `dir`"))?;
    let dir = PathBuf::from(dir);
    let schemas = match p.get("schema") {
        None | Some(Value::Null) => ver::export::schemas_param(None),
        Some(Value::String(v)) => ver::export::schemas_param(Some(v)),
        Some(_) => None,
    }
    .ok_or_else(|| bad(C, "`schema` must be \"both\", \"1.1-draft\" or \"1.0.1\""))?;
    let background = match p.get("background") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(C, "`background` must be true or false"))?,
    };
    let ids = ids_param(s, p, C)?;
    if ids.is_empty() {
        return Err(bad(C, "no photos: give `ids` or select some"));
    }
    // a Record is never written over a library original
    let guard = std::sync::Arc::new(s.original_guard());
    let mut skipped = Vec::new();
    let mut jobs = Vec::new();
    for id in ids {
        match ver::export::job(s, id, &dir, &schemas, &guard) {
            Ok(j) => jobs.push(j),
            Err(e) => skipped.push(json!({"id": id.0, "code": e.code(), "reason": e.to_string()})),
        }
    }
    if background {
        let queued = jobs.len();
        for j in jobs {
            s.ver.queue.push(j);
        }
        return Ok(json!({"queued": queued, "skipped": skipped, "dir": dir.to_string_lossy(), "background": true}));
    }
    // now, with a row in the activity stack while it runs
    let total = jobs.len();
    let task = s.activity.start("ver", "Writing VER records", crate::activity::Cancel::No);
    let (mut written, mut files) = (0usize, Vec::new());
    for (i, j) in jobs.into_iter().enumerate() {
        task.progress(i as u64, total as u64);
        let d = ver::export::run(j, &|path, bytes| crate::export::write_file(path, bytes));
        ver::export::absorb(s, &d);
        match d.result {
            Ok((paths, _, _)) => {
                written += 1;
                files.extend(paths);
            }
            Err(e) => skipped.push(json!({"id": d.photo.0, "code": e.code(), "reason": e.to_string()})),
        }
    }
    drop(task);
    if let Err(e) = s.vera.digests.save() {
        log::warn!("ver: could not save the picture digests: {e}");
    }
    Ok(json!({"written": written, "files": files, "skipped": skipped, "dir": dir.to_string_lossy()}))
}

/// `ver.import {path? | record?}`.
fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "ver.import";
    let text = match (p.get("path").filter(|v| !v.is_null()), p.get("record").filter(|v| !v.is_null())) {
        (Some(_), Some(_)) => return Err(bad(C, "give `path` or `record`, not both")),
        (Some(Value::String(path)), None) => {
            use std::io::Read;
            let mut t = String::new();
            std::fs::File::open(path)
                .and_then(|f| f.take(ver::consume::MAX_RECORD_BYTES as u64 + 1).read_to_string(&mut t))
                .map_err(|e| bad(C, format!("{path}: {e}")))?;
            t
        }
        (None, Some(Value::String(t))) => t.clone(),
        (None, Some(v @ Value::Object(_))) => v.to_string(),
        (Some(_), None) => return Err(bad(C, "`path` must be a file path")),
        (None, Some(_)) => return Err(bad(C, "`record` must be a VER Record (an object, or its JSON text)")),
        (None, None) => return Err(bad(C, "missing `path` or `record`")),
    };
    let out = ver::consume::consume(s, &text).map_err(|e| ver_err(C, e))?;
    Ok(out.to_json())
}

/// `ver.status {}`.
fn status(s: &mut Session, _: &Value) -> Result<Value> {
    let mut reasons = Vec::new();
    // cheap: nothing is loaded or hashed to answer
    let model = match ver::readiness(s) {
        Ok((space_id, weights)) => json!({"ready": true, "spaceId": space_id, "weightsSha256": weights}),
        Err(e) => {
            reasons.push(e.to_string());
            json!({"ready": false, "code": e.code(), "reason": e.to_string()})
        }
    };
    let prefs = s.ver.prefs.clone();
    let signing = match &prefs.sign_key_path {
        None => json!({"enabled": false}),
        #[cfg(not(target_arch = "wasm32"))]
        Some(path) => match ver::sign::Signer::load(std::path::Path::new(path)) {
            Ok(k) => json!({"enabled": true, "keyId": k.key_id, "path": path}),
            Err(e) => {
                reasons.push(format!("signing is configured but the key cannot be used: {e}"));
                json!({"enabled": true, "error": e, "path": path})
            }
        },
        #[cfg(target_arch = "wasm32")]
        Some(path) => json!({"enabled": true, "path": path, "error": "signing needs the desktop app"}),
    };
    Ok(json!({
        "ready": reasons.is_empty(),
        "reasons": reasons,
        "authority": prefs.authority(),
        "generator": {"name": ver::GENERATOR, "version": env!("CARGO_PKG_VERSION")},
        "producerId": prefs.producer_id,
        "defaultSchema": ver::schema_name(prefs.default_schema()),
        "exportSidecars": prefs.export_sidecars(),
        "signing": signing,
        "model": model,
        "lastProduced": s.ver.last_produced,
        "background": {"pending": s.ver.queue.pending()},
        "claims": {
            "reference": false,
            "note": "LightCraft's space declares no reference set: it makes no conformance claim and its vectors are comparable only within this producer (VER 1.1 draft §8.2). CPNP-1 identity is PROVISIONAL until a second decoder evaluation is published (§8.3).",
        },
    }))
}

/// `ver.config {authority?, exportSidecars?, signKeyPath?, defaultSchema?}`.
fn config(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "ver.config";
    let mut prefs = s.ver.prefs.clone();
    if let Some(v) = p.get("authority") {
        prefs.authority = match v {
            Value::Null => None,
            Value::String(a) if ver::space::valid_authority(a.trim()) => Some(a.trim().to_string()),
            _ => {
                return Err(bad(
                    C,
                    "`authority` must be a lowercase reverse-DNS name you control, e.g. \"org.verstandard\" or \"com.example.photos\" (at least two labels; no hyphen in the first)",
                ));
            }
        };
    }
    if let Some(v) = p.get("exportSidecars") {
        prefs.export_sidecars = match v {
            Value::Null => None,
            Value::Bool(b) => Some(*b),
            _ => return Err(bad(C, "`exportSidecars` must be true or false")),
        };
    }
    if let Some(v) = p.get("signKeyPath") {
        prefs.sign_key_path = match v {
            Value::Null => None,
            Value::String(path) if !path.trim().is_empty() => {
                #[cfg(not(target_arch = "wasm32"))]
                ver::sign::Signer::load(std::path::Path::new(path.trim())).map_err(|e| bad(C, e))?;
                Some(path.trim().to_string())
            }
            _ => return Err(bad(C, "`signKeyPath` must be the path of an Ed25519 PKCS#8 PEM, or null")),
        };
    }
    if let Some(v) = p.get("defaultSchema") {
        prefs.default_schema = match v {
            Value::Null => None,
            Value::String(t) if ver::parse_schema(t).is_some() => ver::parse_schema(t).map(|s| ver::schema_name(s).to_string()),
            _ => return Err(bad(C, "`defaultSchema` must be \"1.1-draft\" or \"1.0.1\"")),
        };
    }
    if prefs != s.ver.prefs {
        s.ver.prefs = prefs;
        s.save_prefs()?;
    }
    let pr = &s.ver.prefs;
    Ok(json!({
        "authority": pr.authority(),
        "exportSidecars": pr.export_sidecars(),
        "signKeyPath": pr.sign_key_path,
        "defaultSchema": ver::schema_name(pr.default_schema()),
    }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "ver.record",
            "VER Record",
            [],
            None,
            "{id?: photo (the active one), schema?: \"1.1-draft\" | \"1.0.1\" (ver.config's default, 1.1-draft), level?: L0..L3 (L3)} → {record, issues: [{code, severity, pointer, message}], conformant, level, notes, schema} — the photo's VER Record: CPNP-1 identity (content_hash, pixel_hash, PDQ, pHash), byte-exact metadata segments (L1), the reconciled IPTC view with a trust tier per field (L2), acquisition, custody chain and C2PA status (L3), and its canonical visual embedding in LightCraft's space (no reference set: no conformance claim). Validated (schema + profile) before it is returned; an error says why there is none: no_asset (a demo photo), no_cpnp_path (a camera raw: no Record of its own), needs_model (the embedding model is not installed), not_conformant",
            always,
            record
        ),
        cmd!(
            query "ver.export",
            "Write VER Records",
            [],
            None,
            "{ids?: [photo ids] (the selection), dir, schema?: \"both\" (default) | \"1.1-draft\" | \"1.0.1\", background?: bool (false)} → {written, files, skipped: [{id, code, reason}], dir} — writes each photo's Record to `dir` as <content_hash>.ver.json (and its VER 1.0 twin <content_hash>.ver-1.0.json with \"both\"), under a \"Writing VER records\" activity row; with `background` the photos are queued and written one at a time as vera.pump runs (→ {queued, skipped, dir, background}); at most 10000 photos",
            always,
            export
        ),
        cmd!(
            query "ver.import",
            "Read VER Record",
            [],
            None,
            "{path?: a .ver.json file | record?: a Record (object or JSON text)} → {status: accepted | rejected, accepted, issues, matches: [{id, by: content | pixel_hash | pdq | phash | vector, score}], recordSummary, pendingDigests} — validates the Record under its own schema (schema + profile; an error-severity issue rejects it and nothing is matched), then matches it against the library: the same bytes, the same rendered pixels (a grouping, never an identity), perceptually (score = differing bits) and, in LightCraft's own space only, by vector (score = cosine). Nothing in the library changes",
            always,
            import
        ),
        cmd!(
            query "ver.status",
            "VER Status",
            [],
            None,
            "{} → {ready, reasons, authority, generator, producerId, defaultSchema, exportSidecars, signing: {enabled, keyId?, error?}, model: {ready, spaceId?, …}, lastProduced, background: {pending}, claims} — whether Records can be produced here and now, and with what",
            always,
            status
        ),
        cmd!(
            query "ver.config",
            "VER Settings",
            [],
            None,
            "{authority?: reverse-DNS | null, exportSidecars?: bool | null, signKeyPath?: path | null, defaultSchema?: \"1.1-draft\" | \"1.0.1\" | null} → {authority, exportSidecars, signKeyPath, defaultSchema} — saved in the library's prefs.json; null restores a default (authority org.verstandard, export sidecars on, no signing, 1.1-draft); a signing key is checked when it is set",
            always,
            config
        ),
    ]
}
