//! The vector lane's model, as commands: `vera.model.status`, `vera.model.download`, `vera.model.cancel`,
//! `vera.model.remove`, and `vera.config` (whether the lane runs for this library). Shaped like `segment.model.*`.
//!
//! The model is not part of LightCraft: the user downloads it (SigLIP 2 Base, Apache License 2.0) only after agreeing to
//! its licence, from the pinned mirrors in `lightcraft_embed::fetch` (each file checked by size and SHA-256), or puts
//! its files in the model folder themselves.

// the web build runs none of the lane: what only the desktop uses is unused there
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_variables, unused_mut))]

use serde_json::{Value, json};

use super::lane::{self, LaneState, MODEL_ID};
use crate::cmd::{CommandSpec, always, bad, cmd};
use crate::{EngineError, Result, Session};

/// The model's licence, for the dialog shown before the download.
fn licence() -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    return json!({
        "name": lightcraft_embed::fetch::LICENSE_NAME,
        "url": lightcraft_embed::fetch::LICENSE_URL,
        "notice": lightcraft_embed::fetch::LICENSE_NOTICE,
    });
    #[cfg(target_arch = "wasm32")]
    json!({"name": "Apache License 2.0", "url": "https://huggingface.co/google/siglip2-base-patch32-256", "notice": null})
}

/// `vera.model.status {}`.
fn status(s: &mut Session, _: &Value) -> Result<Value> {
    let mut out = json!({
        "installed": lane::installed(s),
        "modelId": MODEL_ID,
        "bytes": lane::download_bytes(),
        "licence": licence(),
        "loaded": s.vera.lane.engine.loaded(),
        "state": lane::state(s).name(),
    });
    if let Some(d) = &s.vera_model_dir {
        out["dir"] = json!(d.to_string_lossy());
    }
    if let Some(d) = lane::download_json(s) {
        out["downloading"] = d;
    }
    if let Some(e) = lane::download_error(s) {
        out["downloadError"] = json!(e);
    }
    Ok(out)
}

/// `vera.model.download {acceptLicense: true}`: fetch the model in the background (an activity row with ✕).
fn download(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "vera.model.download";
    if p.get("acceptLicense").and_then(Value::as_bool) != Some(true) {
        let lic = licence();
        return Err(bad(
            C,
            format!(
                "the SigLIP 2 model ({MODEL_ID}) is a {:.1} GB download under the {} ({}): pass `acceptLicense: true` once the user has accepted it",
                lane::download_bytes() as f64 / 1e9,
                lic["name"].as_str().unwrap_or("Apache License 2.0"),
                lic["url"].as_str().unwrap_or_default(),
            ),
        ));
    }
    #[cfg(target_arch = "wasm32")]
    {
        Err(bad(C, LaneState::NeedsDesktop.reason()))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = s.vera_model_dir.clone().ok_or_else(|| bad(C, LaneState::NeedsModel(None).reason()))?;
        if lane::installed(s) {
            return Ok(json!({"started": false, "installed": true, "downloading": false}));
        }
        let env = std::env::var(lightcraft_embed::fetch::MIRRORS_ENV).ok();
        let mirrors = lightcraft_embed::fetch::mirrors(env.as_deref(), lane::mirrors_file(s).as_deref());
        if mirrors.is_empty() {
            return Err(bad(C, lightcraft_embed::fetch::no_mirrors_message()));
        }
        let started = s
            .vera
            .lane
            .engine
            .download
            .start(lightcraft_embed::fetch::SIGLIP2_FILES, mirrors, dir, lightcraft_embed::fetch::options(), &s.activity, "SigLIP 2")
            .map_err(EngineError::Other)?;
        // whatever failed before is tried afresh once the files are in place
        s.vera.lane.engine.error = None;
        Ok(json!({"started": started, "installed": false, "downloading": s.vera.lane.engine.download.running()}))
    }
}

/// `vera.model.cancel {}`: stop the download (what arrived is kept and resumed later).
fn cancel(s: &mut Session, _: &Value) -> Result<Value> {
    Ok(json!({"cancelled": lane::cancel_download(s)}))
}

/// `vera.model.remove {}`: delete the model's files from the model folder and this library's embeddings.
fn remove(s: &mut Session, _: &Value) -> Result<Value> {
    super::ensure_open(s);
    lane::cancel_download(s);
    s.vera.lane.unload();
    let mut files = 0usize;
    let mut errors = Vec::new();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dir) = s.vera_model_dir.clone() {
        for f in lightcraft_embed::fetch::SIGLIP2_FILES {
            for path in [dir.join(f.name), dir.join(format!("{}.part", f.name))] {
                match std::fs::remove_file(&path) {
                    Ok(()) => files += 1,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => errors.push(format!("{}: {e}", path.display())),
                }
            }
        }
        // the folder goes too when nothing else is in it
        let _ = std::fs::remove_dir(&dir);
    }
    let sidecar = match s.vera.lane.store.delete_file() {
        Ok(had) => had,
        Err(e) => {
            errors.push(format!("the embeddings beside the library: {e}"));
            false
        }
    };
    s.vera.lane.skipped.clear();
    s.vera.lane.requeue();
    if !errors.is_empty() {
        return Err(EngineError::Other(format!("could not remove everything: {}", errors.join("; "))));
    }
    Ok(json!({"removed": files > 0 || sidecar, "files": files, "sidecar": sidecar, "installed": lane::installed(s)}))
}

/// What `vera.config` reports.
fn config_json(s: &Session) -> Value {
    json!({"vectorLane": s.vera.lane.wanted(), "state": lane::state(s).name()})
}

/// `vera.config {vectorLane?}`: switch the vector lane on or off for this library (saved with its preferences).
fn config(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "vera.config";
    match p.get("vectorLane") {
        None | Some(Value::Null) => {}
        Some(v) => {
            let on = v.as_bool().ok_or_else(|| bad(C, "`vectorLane` must be true or false"))?;
            if s.vera.lane.vector_lane != Some(on) {
                s.vera.lane.vector_lane = Some(on);
                if !on {
                    // the model's memory is given back; the vectors made so far stay
                    s.vera.lane.unload();
                }
                s.save_prefs()?;
            }
        }
    }
    Ok(config_json(s))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "vera.model.status",
            "Search Model Status",
            [],
            None,
            "{} → {installed, dir?, downloading?: {done, total, file}, downloadError?, licence: {name, url, notice}, modelId, bytes, loaded, state} — whether the SigLIP 2 model behind finding photos by meaning is installed (and where), its download's progress, and the licence to show before downloading it",
            always,
            status
        ),
        cmd!(
            query "vera.model.download",
            "Download Search Model",
            [],
            None,
            "{acceptLicense: true} → {started, installed, downloading} — download the SigLIP 2 model (about 1.5 GB, Apache License 2.0, not LightCraft's) in the background from the pinned mirrors, each file checked; only after the user accepted its licence (vera.model.status has the text). Shown in the activity stack (its ✕ or vera.model.cancel stops it; it resumes later)",
            always,
            download
        ),
        cmd!(
            query "vera.model.cancel",
            "Cancel Search Model Download",
            [],
            None,
            "{} → {cancelled} — stop the SigLIP 2 download (what has arrived is kept, and a new download resumes from it)",
            always,
            cancel
        ),
        cmd!(
            query "vera.model.remove",
            "Remove Search Model",
            [],
            None,
            "{} → {removed, files, sidecar, installed} — delete the SigLIP 2 model's files from the model folder and this library's photo embeddings (finding photos by meaning stops until it is downloaded again)",
            always,
            remove
        ),
        cmd!(
            query "vera.config",
            "Search Settings",
            [],
            None,
            "{vectorLane?: bool} → {vectorLane, state} — whether finding photos by meaning (the vector lane) runs for this library; on by default once the model is installed. Saved with the library's preferences",
            always,
            config
        ),
    ]
}
