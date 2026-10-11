//! VER Records in the app (`docs/ver.md`): Settings ▸ Search ▸ VER Records (`ver.status`, `ver.config`), Photo ▸ Show
//! VER Record…, File ▸ Write VER Records… and File ▸ Read VER Record…, and the export dialog's "Write VER Records".
//!
//! A Record takes seconds to make at 24 MP (the full-size Canonical Buffer is decoded, fingerprinted and embedded), so
//! the window never waits for one: Show VER Record queues the photo on the engine's own VER thread (`ver.export
//! {background: true}` into a folder of its own, which `vera.pump` works through at the app's pace), shows "Writing
//! the Record…" meanwhile, and reads the Record back once the pump says it is written; `ver.import` then validates it
//! (the same schema and profile checks `ver.record` runs) for the dialog's issues. Write VER Records goes the same way,
//! for the selection, under the engine's "Writing VER records" activity row. The browser build has no threads: there
//! `ver.record` runs inline.

use std::path::{Path, PathBuf};

use egui::RichText;
use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use super::settings::{check, heading, hint};
use crate::LightcraftApp;
use crate::state::Dialog;
use crate::theme::Tokens;
use crate::widgets::register;

/// How long a read of `ver.status` is reused while Settings ▸ Search is open.
const REFRESH_SECS: f64 = 1.0;
/// The most of a Record's JSON the dialog lays out (one with its metadata segments can run to megabytes): Copy JSON
/// and Save… have all of it.
const SHOWN_JSON: usize = 256 * 1024;
/// Most photos one `ver.export` call takes (the engine's cap).
const MAX_IDS: usize = 10_000;
/// The schemas Settings offers as the default for `ver.record`: (`ver.config` value, label).
pub const SCHEMAS: [(&str, &str); 2] = [("1.1-draft", "VER 1.1 (draft)"), ("1.0.1", "VER 1.0.1")];

/// A Show VER Record in flight: the photo's Record is being written by the engine's VER thread into `dir`. (The
/// browser build makes it inline, never in flight.)
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct RecordJob {
    photo: u64,
    dir: PathBuf,
    schema: String,
    /// The engine's list of background failures (`vera.pump` → `ver.failed`) had this many entries when it started:
    /// a failure for this photo after them is this job's.
    failed_from: usize,
}

/// A Write VER Records in flight.
pub struct Batch {
    ids: Vec<u64>,
    dir: String,
    queued: usize,
    /// Photos `ver.export` refused at once (a demo photo, …): `{id, code, reason}`.
    skipped: Vec<Value>,
    failed_from: usize,
}

/// What Show VER Record shows.
#[derive(Clone, Debug, Default)]
pub struct RecordView {
    pub photo: u64,
    pub name: String,
    /// `ver.record`'s result: `{record, issues, conformant, level, notes, schema}`.
    pub result: Option<Value>,
    /// Why there is no Record: (`ver.record`'s error code, the engine's sentence).
    pub error: Option<(String, String)>,
    /// The Record as pretty JSON (what Copy JSON copies and Save… writes).
    pub json: String,
}

impl RecordView {
    pub fn running(&self) -> bool {
        self.result.is_none() && self.error.is_none()
    }
}

/// VER in the app (`Caches::ver`).
#[derive(Default)]
pub struct State {
    pub record_job: Option<RecordJob>,
    pub view: Option<RecordView>,
    pub batch: Option<Batch>,
    /// The last Write VER Records' skipped photos: (file name, reason in the app's words).
    pub skipped: Vec<(String, String)>,
    /// The pump's last `ver` report (`{pending, written, failed}`).
    last: Value,
    /// Folders of Show VER Record requests that were replaced before they finished, removed once the queue is empty.
    stale: Vec<PathBuf>,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    seq: u64,
}

/// Whether a Record is being written for the app (the headless harness waits for it).
pub fn busy(app: &LightcraftApp) -> bool {
    app.caches.ver.record_job.is_some() || app.caches.ver.batch.is_some()
}

/// A `ver.*` error's code: the engine ends its sentence with it (`… (needs_model)`).
pub(crate) fn error_code(e: &str) -> &'static str {
    const CODES: [&str; 7] = ["no_asset", "no_cpnp_path", "needs_model", "not_conformant", "source", "model", "invalid"];
    let tail = e.trim_end().strip_suffix(')').and_then(|r| r.rsplit_once('(')).map(|(_, c)| c).unwrap_or("");
    CODES.iter().copied().find(|c| *c == tail).unwrap_or("invalid")
}

/// The engine's sentence without its "invalid parameters for `…`: " opening and its `(code)` ending.
fn engine_sentence(e: &str) -> String {
    let e = super::vera_settings::engine_message(e).trim_end();
    let code = error_code(e);
    let e = e.strip_suffix(&format!("({code})")).unwrap_or(e).trim_end();
    e.to_string()
}

/// Why a photo has no Record, in the app's words (`code` from `ver.record` / `ver.export`, `reason` the engine's).
pub fn reason_text(code: &str, reason: &str) -> String {
    match code {
        "no_asset" => crate::i18n::tr("A generated demo photo: it has no file, so it has no Record.").to_string(),
        "no_cpnp_path" => crate::i18n::tr("This is a camera raw. It has no Record of its own; its exports do.").to_string(),
        "needs_model" => crate::i18n::tr(
            "A Record carries the picture's embedding, which needs the search-by-meaning model: a one-time download in Settings → Search.",
        )
        .to_string(),
        "not_conformant" => crate::i18n::tr("The Record did not pass its own checks, so LightCraft did not hand it out.").to_string(),
        "source" => crate::i18n::tr_format!("The photo's file could not be read: {e}", e = reason),
        "model" => crate::i18n::tr_format!("The search-by-meaning model did not work: {e}", e = reason),
        _ => super::faces::sentence(reason),
    }
}

// ------------------------------------------------------------------------------------------ Show VER Record

/// The schema `ver.record` uses when none is named (Settings ▸ Search ▸ VER Records ▸ Default schema).
fn default_schema(app: &mut LightcraftApp) -> String {
    let v = app.session.execute("ver.config", &json!({})).unwrap_or(Value::Null);
    v["defaultSchema"].as_str().unwrap_or("1.1-draft").to_string()
}

/// Photo ▸ Show VER Record… (`photo.showVerRecord {id?}`): opens the dialog at once, saying "Writing the Record…", and
/// starts making the Record of photo `id` (the active one) off the UI thread.
pub fn show_record(app: &mut LightcraftApp, ctx: &egui::Context, id: Option<PhotoId>) -> Result<Value, String> {
    let id = id.or_else(|| app.session.active()).ok_or_else(|| crate::i18n::tr("No photo selected").to_string())?;
    let name = app.session.catalog.photo(id).map(|p| p.file_name.clone()).ok_or_else(|| format!("no photo {}", id.0))?;
    let schema = default_schema(app);
    app.caches.ver.view = Some(RecordView { photo: id.0, name, ..Default::default() });
    app.ui.dialog = Some(Dialog::VerRecord { photo: id.0 });
    ctx.request_repaint();
    start_record(app, id, &schema);
    Ok(json!({"photo": id.0, "background": app.caches.ver.record_job.is_some()}))
}

/// Make photo `id`'s Record in `schema`: queued on the engine's VER thread (the result comes back through [`pumped`]).
#[cfg(not(target_arch = "wasm32"))]
fn start_record(app: &mut LightcraftApp, id: PhotoId, schema: &str) {
    let st = &mut app.caches.ver;
    if let Some(old) = st.record_job.take() {
        st.stale.push(old.dir);
    }
    st.seq += 1;
    let dir = std::env::temp_dir().join(format!("lightcraft-ver-record-{}-{}", std::process::id(), st.seq));
    let failed_from = st.last["failed"].as_array().map_or(0, Vec::len);
    let params = json!({"ids": [id.0], "dir": dir.to_string_lossy(), "schema": schema, "background": true});
    match app.session.execute("ver.export", &params) {
        Ok(v) => match v["skipped"].as_array().and_then(|s| s.first()) {
            // refused at once (a demo photo has no file)
            Some(s) => {
                let (code, reason) = (s["code"].as_str().unwrap_or("invalid"), s["reason"].as_str().unwrap_or(""));
                record_failed(app, id.0, code, reason);
            }
            None => {
                app.caches.ver.record_job = Some(RecordJob { photo: id.0, dir, schema: schema.to_string(), failed_from });
                // the pump takes the job on its next call
                app.caches.vera_next_pump = 0.0;
            }
        },
        Err(e) => {
            let e = e.to_string();
            record_failed(app, id.0, error_code(&e), &engine_sentence(&e));
        }
    }
}

/// The browser build has no threads to make it on (and no files to write it to): `ver.record` runs inline.
#[cfg(target_arch = "wasm32")]
fn start_record(app: &mut LightcraftApp, id: PhotoId, schema: &str) {
    match app.session.execute("ver.record", &json!({"id": id.0, "schema": schema})) {
        Ok(v) => record_done(app, id.0, v),
        Err(e) => {
            let e = e.to_string();
            record_failed(app, id.0, error_code(&e), &engine_sentence(&e));
        }
    }
}

fn record_failed(app: &mut LightcraftApp, photo: u64, code: &str, reason: &str) {
    if let Some(v) = app.caches.ver.view.as_mut().filter(|v| v.photo == photo) {
        v.error = Some((code.to_string(), reason.to_string()));
    }
}

/// The Record of `photo` is here (`ver.record`'s shape): shown when its dialog still waits for it.
pub(crate) fn record_done(app: &mut LightcraftApp, photo: u64, result: Value) {
    let Some(v) = app.caches.ver.view.as_mut().filter(|v| v.photo == photo) else { return };
    v.json = serde_json::to_string_pretty(&result["record"]).unwrap_or_default();
    v.result = Some(result);
}

/// The Record file the VER thread wrote into `dir` (it writes atomically: a file under its final name is whole).
fn written_record(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".ver.json") && !n.starts_with('.')))
}

/// A Record read back from its file: validated by `ver.import` (schema, profile and the core's consistency rules, as
/// `ver.record` validates it) and put in `ver.record`'s shape.
fn read_back(app: &mut LightcraftApp, path: &Path, schema: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let record: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let checked = app.session.execute("ver.import", &json!({"record": text})).map_err(|e| e.to_string())?;
    let level = record["conformance_level"].clone();
    Ok(json!({
        "record": record,
        "issues": checked["issues"],
        "conformant": checked["accepted"] == true,
        "level": level,
        "notes": [],
        "schema": schema,
    }))
}

/// After each `vera.pump` (`ver`: its `{pending, written, failed}`): a Show VER Record whose Record is written (or
/// failed) is shown, and a Write VER Records that has finished says so.
pub fn pumped(app: &mut LightcraftApp, ctx: &egui::Context, ver: &Value) {
    if ver.is_object() {
        app.caches.ver.last = ver.clone();
    }
    let failed: Vec<Value> = ver["failed"].as_array().cloned().unwrap_or_default();
    let pending = ver["pending"].as_u64().unwrap_or(0);
    if let Some(job) = app.caches.ver.record_job.as_ref() {
        let (photo, dir, schema) = (job.photo, job.dir.clone(), job.schema.clone());
        let failure = failed.iter().skip(job.failed_from).find(|f| f["id"].as_u64() == Some(photo)).cloned();
        if let Some(path) = written_record(&dir) {
            app.caches.ver.record_job = None;
            match read_back(app, &path, &schema) {
                Ok(v) => record_done(app, photo, v),
                Err(e) => record_failed(app, photo, "invalid", &e),
            }
            let _ = std::fs::remove_dir_all(&dir);
            ctx.request_repaint();
        } else if let Some(f) = failure {
            app.caches.ver.record_job = None;
            record_failed(app, photo, f["code"].as_str().unwrap_or("invalid"), f["reason"].as_str().unwrap_or(""));
            let _ = std::fs::remove_dir_all(&dir);
            ctx.request_repaint();
        } else if pending == 0 {
            // neither written nor failed, and nothing left to do: the VER thread went away
            app.caches.ver.record_job = None;
            record_failed(app, photo, "invalid", crate::i18n::tr("The Record could not be written."));
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    if pending == 0 {
        for d in std::mem::take(&mut app.caches.ver.stale) {
            let _ = std::fs::remove_dir_all(d);
        }
        if app.caches.ver.batch.is_some() {
            batch_done(app, ctx, &failed);
        }
    }
}

// ------------------------------------------------------------------------------------------ Write VER Records

/// File ▸ Write VER Records… (`library.writeVerRecords {dir, ids?}`): the selection's Records (every photo in view
/// when nothing is selected) into `dir`, as `<content_hash>.ver.json` and its VER 1.0 twin, written one at a time on
/// the engine's VER thread under its "Writing VER records" row. When the last is written a toast gives the counts, with
/// "Show skipped" when some photos have no Record.
pub fn write_records(app: &mut LightcraftApp, ctx: &egui::Context, dir: &str, ids: Option<Vec<u64>>) -> Result<Value, String> {
    if app.caches.ver.batch.is_some() {
        return Err(crate::i18n::tr("VER Records are being written already").to_string());
    }
    let ids: Vec<u64> = match ids {
        Some(ids) => ids,
        None if !app.session.selection.ids.is_empty() => app.session.selection.ids.iter().map(|id| id.0).collect(),
        None => app.session.visible_cloned().iter().map(|id| id.0).collect(),
    };
    if ids.is_empty() {
        return Err(crate::i18n::tr("No photos to write Records for").to_string());
    }
    let failed_from = app.caches.ver.last["failed"].as_array().map_or(0, Vec::len);
    let (mut queued, mut skipped) = (0usize, Vec::new());
    for chunk in ids.chunks(MAX_IDS) {
        let r =
            app.session.execute("ver.export", &json!({"ids": chunk, "dir": dir, "background": true})).map_err(|e| engine_sentence(&e.to_string()));
        let r = match r {
            Ok(r) => r,
            // a later chunk refused: what was queued is still written
            Err(e) if queued > 0 => {
                log::warn!("ver.export: {e}");
                break;
            }
            Err(e) => return Err(e),
        };
        queued += r["queued"].as_u64().unwrap_or(0) as usize;
        skipped.extend(r["skipped"].as_array().cloned().unwrap_or_default());
    }
    app.caches.ver.batch = Some(Batch { ids: ids.clone(), dir: dir.to_string(), queued, skipped, failed_from });
    if queued == 0 {
        batch_done(app, ctx, &[]);
    } else {
        app.caches.vera_next_pump = 0.0;
        app.toast(ctx, crate::i18n::tr_format!("Writing VER Records for {n} photos…", n = queued));
    }
    Ok(json!({"queued": queued, "photos": ids.len(), "dir": dir}))
}

/// A Write VER Records has nothing left to write: the toast with its counts (and "Show skipped").
fn batch_done(app: &mut LightcraftApp, ctx: &egui::Context, failed: &[Value]) {
    let Some(b) = app.caches.ver.batch.take() else { return };
    let mut skipped = b.skipped;
    for f in failed.iter().skip(b.failed_from) {
        let id = f["id"].as_u64().unwrap_or(0);
        if b.ids.contains(&id) && !skipped.iter().any(|s| s["id"].as_u64() == Some(id)) {
            skipped.push(f.clone());
        }
    }
    let late = skipped.len().saturating_sub(b.ids.len().saturating_sub(b.queued));
    let written = b.queued.saturating_sub(late);
    let rows: Vec<(String, String)> = skipped
        .iter()
        .map(|s| {
            let id = s["id"].as_u64().unwrap_or(0);
            let name = app.session.catalog.photo(PhotoId(id)).map_or_else(|| format!("#{id}"), |p| p.file_name.clone());
            (name, reason_text(s["code"].as_str().unwrap_or(""), s["reason"].as_str().unwrap_or("")))
        })
        .collect();
    let mut text =
        crate::i18n::tr_format!("Wrote VER Records for {written} of {total} photos to {dir}", written = written, total = b.ids.len(), dir = b.dir);
    if !rows.is_empty() {
        text.push_str(&crate::i18n::tr_format!(" · {n} skipped", n = rows.len()));
    }
    app.caches.ver.skipped = rows;
    app.ui.status = text.clone();
    app.toast_for(ctx, text, 8.0);
    if !app.caches.ver.skipped.is_empty() {
        super::vera::offer(app, crate::i18n::tr("Show skipped"), "dialog.verSkipped", json!({}));
    }
}

/// The skipped photos' dialog: each file and why it has no Record.
pub fn skipped_dialog(ui: &mut egui::Ui, t: &Tokens, rows: &[(String, String)]) {
    if rows.is_empty() {
        ui.label(RichText::new(crate::i18n::tr("Every photo got its Record.")).color(t.text_label));
        return;
    }
    egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, true]).show(ui, |ui| {
        egui::Grid::new("ver-skipped").num_columns(2).spacing([14.0, 4.0]).striped(true).show(ui, |ui| {
            for (i, (name, reason)) in rows.iter().enumerate() {
                let r = ui.label(RichText::new(name).color(t.text));
                register(ui.ctx(), format!("verSkipped:{i}"), r.rect);
                ui.add(egui::Label::new(RichText::new(reason).font(t.font(11.5)).color(t.text_dim)).wrap());
                ui.end_row();
            }
        });
    });
}

// ------------------------------------------------------------------------------------------ Read VER Record

/// File ▸ Read VER Record… (`file.readVerRecord {path}`): `ver.import` reads the file, validates it and matches it
/// against the library; the dialog says what it found. Nothing in the library changes.
pub fn read_record(app: &mut LightcraftApp, path: &str) -> Result<Value, String> {
    let file = Path::new(path).file_name().map_or_else(|| path.to_string(), |n| n.to_string_lossy().to_string());
    let r = app.session.execute("ver.import", &json!({"path": path}));
    let (result, error) = match &r {
        Ok(v) => (v.clone(), None),
        Err(e) => (Value::Null, Some(super::faces::sentence(&engine_sentence(&e.to_string())))),
    };
    app.ui.dialog = Some(Dialog::VerImport { file, result, error });
    r.map_err(|e| e.to_string())
}

/// What a match says about how it matches.
pub(crate) fn match_group(by: &str) -> &'static str {
    match by {
        "content" => "Same file",
        "pixel_hash" => "Same pixels (a grouping, not identity)",
        _ => "Looks alike",
    }
}

/// A look-alike's score as words: bits apart for the perceptual digests, the cosine for a vector.
fn score_text(by: &str, score: f64) -> String {
    match by {
        "pdq" => crate::i18n::tr_format!("PDQ: {bits} of 256 bits apart", bits = score.round() as i64),
        "phash" => crate::i18n::tr_format!("pHash: {bits} of 64 bits apart", bits = score.round() as i64),
        "vector" => crate::i18n::tr_format!("by meaning: cosine {score:.3}", score = score),
        _ => String::new(),
    }
}

/// The Read VER Record dialog's body.
pub fn import_dialog(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, file: &str, result: &Value, error: Option<&str>) {
    ui.label(RichText::new(file).font(t.font(11.5)).color(t.text_dim));
    if let Some(e) = error {
        let r = ui.add(egui::Label::new(RichText::new(e).color(t.caution)).wrap());
        register(ui.ctx(), "verImport:error", r.rect);
        return;
    }
    let accepted = result["accepted"] == true;
    ui.horizontal_wrapped(|ui| {
        let (text, colour) = if accepted { ("Record accepted", t.accent) } else { ("Record rejected", t.caution) };
        let r = ui.label(RichText::new(crate::i18n::tr(text)).font(t.semibold(13.0)).color(colour));
        register(ui.ctx(), "verImport:status", r.rect);
        let why =
            if accepted { "It passes its schema and the VER checks." } else { "It fails its schema or the VER checks, so nothing was matched." };
        ui.label(RichText::new(crate::i18n::tr(why)).color(t.text_label));
    });
    // the Record, in a few lines
    let s = &result["recordSummary"];
    let text = |v: &Value| match v {
        Value::String(s) => s.clone(),
        Value::Null => "—".to_string(),
        v => v.to_string(),
    };
    egui::Grid::new("ver-import-summary").num_columns(2).spacing([14.0, 3.0]).show(ui, |ui| {
        let row = |ui: &mut egui::Ui, label: &str, value: String| {
            ui.label(RichText::new(crate::i18n::tr(label)).color(t.text_dim));
            ui.add(egui::Label::new(RichText::new(value).font(t.font(11.5)).color(t.text_label)).wrap());
            ui.end_row();
        };
        row(ui, "VER version", text(&s["verVersion"]));
        row(ui, "Conformance level", text(&s["level"]));
        let generator = &s["generator"];
        if generator.is_object() {
            row(ui, "Made by", format!("{} {}", text(&generator["name"]), generator["version"].as_str().unwrap_or("")).trim().to_string());
        }
        row(ui, "Content hash", text(&s["contentHash"]));
        let size = match (s["width"].as_u64(), s["height"].as_u64()) {
            (Some(w), Some(h)) => format!(" · {w} × {h}"),
            _ => String::new(),
        };
        row(ui, "Pixel hash", format!("{}{size}", text(&s["pixelHash"])));
        row(ui, "Format", text(&s["format"]));
        let spaces: Vec<String> = s["spaces"].as_array().into_iter().flatten().map(text).collect();
        row(ui, "Spaces", if spaces.is_empty() { "—".into() } else { spaces.join(", ") });
        row(ui, "Signed", crate::i18n::tr(if s["signed"] == true { "Yes" } else { "No" }).to_string());
    });
    issues(ui, t, &result["issues"], "verImport");
    // the matches, grouped by how they match
    ui.add_space(2.0);
    ui.label(RichText::new(crate::i18n::tr("In this library")).font(t.semibold(12.5)).color(t.text));
    let matches: Vec<&Value> = result["matches"].as_array().map(|m| m.iter().collect()).unwrap_or_default();
    if matches.is_empty() {
        let r =
            ui.label(RichText::new(crate::i18n::tr(if accepted { "No photo here matches it." } else { "Nothing was matched." })).color(t.text_dim));
        register(ui.ctx(), "verImport:noMatches", r.rect);
    }
    egui::ScrollArea::vertical().id_salt("ver-import-matches").max_height(200.0).auto_shrink([false, true]).show(ui, |ui| {
        for group in ["Same file", "Same pixels (a grouping, not identity)", "Looks alike"] {
            let rows: Vec<&&Value> = matches.iter().filter(|m| match_group(m["by"].as_str().unwrap_or("")) == group).collect();
            if rows.is_empty() {
                continue;
            }
            let r = ui.label(RichText::new(crate::i18n::tr(group)).font(t.semibold(11.5)).color(t.text_label));
            register(ui.ctx(), format!("verImport:group:{}", rows[0]["by"].as_str().unwrap_or("")), r.rect);
            for m in rows {
                let id = m["id"].as_u64().unwrap_or(0);
                let name = app.session.catalog.photo(PhotoId(id)).map_or_else(|| format!("#{id}"), |p| p.file_name.clone());
                let by = m["by"].as_str().unwrap_or("");
                let score = score_text(by, m["score"].as_f64().unwrap_or(0.0));
                let line = if score.is_empty() { format!("   {name}") } else { format!("   {name} · {score}") };
                ui.label(RichText::new(line).font(t.font(11.5)).color(t.text));
            }
        }
    });
    let pending = result["pendingDigests"].as_u64().unwrap_or(0);
    if pending > 0 {
        let r = ui.label(
            RichText::new(crate::i18n::tr_format!("{n} photos are still being fingerprinted; matches may be incomplete", n = pending))
                .font(t.font(11.5))
                .color(t.text_dim),
        );
        register(ui.ctx(), "verImport:pending", r.rect);
    }
    let ids: Vec<u64> = matches.iter().filter_map(|m| m["id"].as_u64()).collect();
    let r = ui.add_enabled(!ids.is_empty(), egui::Button::new(crate::i18n::tr("Select Matches")));
    register(ui.ctx(), "button:verSelectMatches", r.rect);
    if r.clicked() {
        match app.run("library.select", json!({"ids": ids})) {
            Ok(_) => app.toast(ui.ctx(), crate::i18n::tr_format!("Selected {n} matching photos", n = ids.len())),
            Err(e) => app.toast_error(ui.ctx(), e),
        }
    }
}

/// A Record's validation issues (`[{code, severity, pointer, message}]`), or that there are none.
fn issues(ui: &mut egui::Ui, t: &Tokens, issues: &Value, widget: &str) {
    let list = issues.as_array().map(Vec::as_slice).unwrap_or_default();
    ui.add_space(2.0);
    ui.label(RichText::new(crate::i18n::tr("Issues")).font(t.semibold(12.5)).color(t.text));
    if list.is_empty() {
        let r = ui.label(RichText::new(crate::i18n::tr("None")).color(t.text_dim));
        register(ui.ctx(), format!("{widget}:noIssues"), r.rect);
        return;
    }
    egui::ScrollArea::vertical().id_salt(format!("{widget}-issues")).max_height(140.0).auto_shrink([false, true]).show(ui, |ui| {
        egui::Grid::new(format!("{widget}-issues-grid")).num_columns(4).spacing([10.0, 3.0]).striped(true).show(ui, |ui| {
            for (i, issue) in list.iter().enumerate() {
                let severity = issue["severity"].as_str().unwrap_or("");
                let colour = if severity.eq_ignore_ascii_case("error") { t.caution } else { t.text_label };
                let r = ui.label(RichText::new(issue["code"].as_str().unwrap_or("")).font(t.semibold(11.5)).color(colour));
                register(ui.ctx(), format!("{widget}:issue:{i}"), r.rect);
                ui.label(RichText::new(severity).font(t.font(11.5)).color(colour));
                ui.label(RichText::new(issue["pointer"].as_str().unwrap_or("")).font(egui::FontId::monospace(11.0)).color(t.text_dim));
                ui.add(egui::Label::new(RichText::new(issue["message"].as_str().unwrap_or("")).font(t.font(11.5)).color(t.text_label)).wrap());
                ui.end_row();
            }
        });
    });
}

// ------------------------------------------------------------------------------------------ the Record dialog

/// The perceptual digest `alg` of a Record (`pdq/1.0`, `phash-dct/64`).
fn perceptual<'a>(record: &'a Value, prefix: &str) -> Option<&'a str> {
    record.pointer("/identity/perceptual")?.as_array()?.iter().find(|p| p["alg"].as_str().is_some_and(|a| a.starts_with(prefix)))?["value"].as_str()
}

/// What a click in the Record dialog asks for (done once the dialog is drawn).
enum Act {
    OpenSettings,
    Save,
}

/// The Show VER Record dialog's body: "Writing the Record…" until it is here, then its badge, identity, space, issues
/// and JSON (Copy JSON, Save…), or why the photo has none.
pub fn record_dialog(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, photo: u64) {
    // (borrowed, not copied: a Record can be megabytes, and this runs every frame)
    let Some(view) = app.caches.ver.view.take() else {
        ui.label(RichText::new(crate::i18n::tr("Nothing to show.")).color(t.text_dim));
        return;
    };
    let act = if view.photo == photo { record_body(ui, t, &view) } else { None };
    app.caches.ver.view = Some(view);
    match act {
        Some(Act::OpenSettings) => app.ui.dialog = Some(Dialog::Settings { tab: "search".into() }),
        Some(Act::Save) => {
            if let Err(e) = app.run("dialog.verRecordSave", json!({})) {
                app.toast_error(ui.ctx(), e);
            }
        }
        None => {}
    }
}

fn record_body(ui: &mut egui::Ui, t: &Tokens, view: &RecordView) -> Option<Act> {
    ui.label(RichText::new(&view.name).font(t.semibold(13.0)).color(t.text));
    if view.running() {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new());
            let r = ui.label(RichText::new(crate::i18n::tr("Writing the Record…")).color(t.text_label));
            register(ui.ctx(), "verRecord:running", r.rect);
        });
        hint(ui, t, "The picture is decoded in full, fingerprinted and embedded: a few seconds for a large photo. The window stays free meanwhile.");
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        return None;
    }
    if let Some((code, reason)) = &view.error {
        let r = ui.add(egui::Label::new(RichText::new(reason_text(code, reason)).color(t.caution)).wrap());
        register(ui.ctx(), "verRecord:error", r.rect);
        // the engine's detail where it adds something (which format, which checks)
        if !reason.is_empty() && matches!(code.as_str(), "no_cpnp_path" | "not_conformant") {
            ui.add(egui::Label::new(RichText::new(super::faces::sentence(reason)).font(t.font(11.5)).color(t.text_dim)).wrap());
        }
        if code == "needs_model" {
            let r = ui.button(crate::i18n::tr("Open Settings → Search"));
            register(ui.ctx(), "button:verOpenSearch", r.rect);
            if r.clicked() {
                return Some(Act::OpenSettings);
            }
        }
        return None;
    }
    let result = view.result.as_ref()?;
    let record = &result["record"];
    // the badge: conformant, its level and schema
    ui.horizontal_wrapped(|ui| {
        let ok = result["conformant"] == true;
        let (text, colour) = if ok { ("Conformant", t.accent) } else { ("Not conformant", t.caution) };
        let r = ui.label(RichText::new(crate::i18n::tr(text)).font(t.semibold(12.5)).color(colour));
        register(ui.ctx(), "verRecord:badge", r.rect);
        let level = result["level"].as_str().unwrap_or("");
        ui.label(RichText::new(crate::i18n::tr_format!("Level {level}", level = level)).font(t.semibold(12.5)).color(t.text));
        let schema = result["schema"].as_str().unwrap_or("");
        let schema = SCHEMAS.iter().find(|s| s.0 == schema).map_or(schema, |s| s.1);
        ui.label(RichText::new(crate::i18n::tr(schema)).color(t.text_dim));
        if record.pointer("/provenance/signature").is_some() {
            ui.label(RichText::new(crate::i18n::tr("signed")).color(t.text_dim));
        }
    });
    // what the picture is
    let mono = egui::FontId::monospace(11.0);
    let pixel = record.pointer("/identity/pixel_hash");
    let size = match pixel.and_then(|p| Some((p["width"].as_u64()?, p["height"].as_u64()?))) {
        Some((w, h)) => format!("{w} × {h}"),
        None => String::new(),
    };
    let space = record.pointer("/embeddings/0/space_id").or_else(|| record.pointer("/spaces/0/space_id")).and_then(Value::as_str);
    let rows: [(&str, &str, Option<&str>, String); 5] = [
        ("contentHash", "Content hash", record.pointer("/identity/content_hash/value").and_then(Value::as_str), String::new()),
        ("pixelHash", "Pixel hash", pixel.and_then(|p| p["value"].as_str()), size),
        ("pdq", "PDQ", perceptual(record, "pdq"), String::new()),
        ("phash", "pHash", perceptual(record, "phash"), String::new()),
        ("space", "Space", space, String::new()),
    ];
    egui::Grid::new("ver-record-identity").num_columns(3).spacing([12.0, 3.0]).show(ui, |ui| {
        for (key, label, value, extra) in rows {
            ui.label(RichText::new(crate::i18n::tr(label)).color(t.text_dim));
            let shown = match value {
                Some(v) if extra.is_empty() => v.to_string(),
                Some(v) => format!("{v} · {extra}"),
                None => "—".to_string(),
            };
            // (a 64-digit hash on one line where the dialog has room)
            let r = ui.vertical(|ui| {
                ui.set_width(450.0);
                ui.add(egui::Label::new(RichText::new(shown).font(mono.clone()).color(t.text_label)).wrap())
            });
            register(ui.ctx(), format!("verRecord:{key}"), r.inner.rect);
            match value {
                Some(v) => {
                    let r = ui.small_button(crate::i18n::tr("Copy"));
                    register(ui.ctx(), format!("button:verCopy:{key}"), r.rect);
                    if r.clicked() {
                        ui.ctx().copy_text(v.to_string());
                    }
                }
                None => {
                    ui.label("");
                }
            }
            ui.end_row();
        }
    });
    issues(ui, t, &result["issues"], "verRecord");
    for note in result["notes"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        ui.add(egui::Label::new(RichText::new(note).font(t.font(11.5)).color(t.text_dim)).wrap());
    }
    // the Record itself (at most SHOWN_JSON of it laid out, cut at a char boundary)
    ui.add_space(2.0);
    ui.label(RichText::new("JSON").font(t.semibold(12.5)).color(t.text));
    let mut end = view.json.len().min(SHOWN_JSON);
    while end > 0 && !view.json.is_char_boundary(end) {
        end -= 1;
    }
    let shown = view.json.get(..end).unwrap_or("");
    let r = egui::ScrollArea::vertical().id_salt("ver-record-json").max_height(220.0).auto_shrink([false, true]).show(ui, |ui| {
        let mut text = shown;
        ui.add(egui::TextEdit::multiline(&mut text).font(mono.clone()).desired_width(f32::INFINITY).desired_rows(8))
    });
    register(ui.ctx(), "verRecord:json", r.inner_rect);
    if shown.len() < view.json.len() {
        hint(ui, t, "Shortened here: Copy JSON and Save… have all of it.");
    }
    let mut act = None;
    ui.horizontal(|ui| {
        let r = ui.button(crate::i18n::tr("Copy JSON"));
        register(ui.ctx(), "button:verCopyJson", r.rect);
        if r.clicked() {
            ui.ctx().copy_text(view.json.clone());
        }
        let r = ui.button(crate::i18n::tr("Save…"));
        register(ui.ctx(), "button:verSave", r.rect);
        if r.clicked() {
            act = Some(Act::Save);
        }
    });
    act
}

/// The file name Save… suggests for the Record shown: `<content_hash>.ver.json`, as `ver.export` names it.
pub fn suggested_name(app: &LightcraftApp) -> String {
    let hash = app
        .caches
        .ver
        .view
        .as_ref()
        .and_then(|v| v.result.as_ref())
        .and_then(|r| r.pointer("/record/identity/content_hash/value")?.as_str().map(str::to_string));
    format!("{}.ver.json", hash.unwrap_or_else(|| "record".into()))
}

/// Save… in the Record dialog, once the save dialog has answered `path`: the Record's JSON, as shown.
pub fn save_record(app: &mut LightcraftApp, ctx: &egui::Context, path: &str) -> Result<Value, String> {
    let json = app.caches.ver.view.as_ref().filter(|v| v.result.is_some()).map(|v| v.json.clone()).ok_or("no Record to save")?;
    app.session.check_write_target(path)?;
    match app.services.write.as_mut() {
        Some(w) => w(path, json.as_bytes())?,
        None => lightcraft_engine::export::write_file(path, json.as_bytes())?,
    }
    app.toast(ctx, crate::i18n::tr_format!("Saved the VER Record as {path}", path = path));
    Ok(json!({"path": path, "bytes": json.len()}))
}

// ------------------------------------------------------------------------------------------ Settings ▸ Search

fn status_id() -> egui::Id {
    egui::Id::new("ver-settings-status")
}

/// `ver.status`, re-read at most every [`REFRESH_SECS`], or at once after an action (`vera_epoch`).
fn read_status(app: &mut LightcraftApp, ctx: &egui::Context) -> Value {
    let now = ctx.input(|i| i.time);
    let epoch = app.caches.vera_epoch;
    if let Some((e, at, v)) = ctx.data(|d| d.get_temp::<(u64, f64, Value)>(status_id()))
        && e == epoch
        && now - at < REFRESH_SECS
    {
        return v;
    }
    let v = app.session.execute("ver.status", &json!({})).unwrap_or(Value::Null);
    ctx.data_mut(|d| d.insert_temp(status_id(), (epoch, now, v.clone())));
    v
}

/// The section's first line: whether Records can be made here, and what they lack. `true` with it when the model is
/// what they lack.
pub(crate) fn status_line(status: &Value) -> (String, bool, bool) {
    let signing = &status["signing"];
    if status["ready"] == true {
        let text = match signing["keyId"].as_str() {
            Some(key) => crate::i18n::tr_format!("Ready · signed as {key}", key = key),
            None => crate::i18n::tr("Ready · unsigned").to_string(),
        };
        return (text, true, false);
    }
    if status["model"]["ready"] == false && status["model"]["code"] == "needs_model" {
        return (crate::i18n::tr("Needs the model: Records carry an embedding").to_string(), false, true);
    }
    if let Some(e) = signing["error"].as_str() {
        return (crate::i18n::tr_format!("The signing key can't be used: {e}", e = e), false, false);
    }
    let reasons: Vec<&str> = status["reasons"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let text = if reasons.is_empty() { crate::i18n::tr("Not available in this build").to_string() } else { reasons.join(" · ") };
    (text, false, false)
}

/// What the Authority field holds while it is edited (loaded from `ver.status` when the section appears, or after an
/// action).
#[derive(Clone, Default)]
struct Form {
    pass: u64,
    epoch: u64,
    authority: String,
    error: Option<String>,
}

fn form_id() -> egui::Id {
    egui::Id::new("ver-settings-form")
}

/// Apply `params` through `ver.config`: the engine's refusal goes to the section, in a sentence.
fn configure(app: &mut LightcraftApp, form: &mut Form, params: Value) {
    match app.session.execute("ver.config", &params) {
        Ok(_) => form.error = None,
        Err(e) => form.error = Some(super::faces::sentence(super::vera_settings::engine_message(&e.to_string()))),
    }
    app.caches.vera_epoch += 1;
    // applied: the form reloads from what the engine kept; refused: it keeps what was typed, with the error
    form.epoch = if form.error.is_none() { 0 } else { app.caches.vera_epoch };
}

/// Settings ▸ Search ▸ VER Records: whether Records can be made (the model, the signing key), and the authority, the
/// export sidecars, the signing key and the default schema (`ver.config`).
pub fn settings_section(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, model: &Value) {
    heading(ui, t, "VER Records");
    let ctx = ui.ctx().clone();
    let status = read_status(app, &ctx);
    if status.is_null() {
        hint(ui, t, "VER Records are not available in this build.");
        return;
    }
    let pass = ctx.cumulative_pass_nr();
    let epoch = app.caches.vera_epoch;
    let mut form = ctx.data(|d| d.get_temp::<Form>(form_id())).unwrap_or_default();
    if form.pass.checked_add(1) != Some(pass) || form.epoch != epoch {
        form = Form { authority: status["authority"].as_str().unwrap_or("").to_string(), epoch, ..Default::default() };
    }
    form.pass = pass;
    let (line, ready, needs_model) = status_line(&status);
    ui.horizontal_wrapped(|ui| {
        let colour = if ready { t.accent } else { t.caution };
        let r = ui.add(egui::Label::new(RichText::new(line).font(t.semibold(11.5)).color(colour)).wrap());
        register(ui.ctx(), "ver:status", r.rect);
        if needs_model && model["dir"].is_string() && model["installed"] != true && model["downloading"].is_null() {
            let r = ui.link(crate::i18n::tr("Download the model…"));
            register(ui.ctx(), "link:verModel", r.rect);
            if r.clicked() {
                super::vera_settings::open_licence(app, model);
            }
        }
    });
    let mut sidecars = status["exportSidecars"] != false;
    let schema = status["defaultSchema"].as_str().unwrap_or("1.1-draft").to_string();
    let signing = status["signing"].clone();
    let mut apply = false;
    egui::Grid::new("ver-settings").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label(RichText::new(crate::i18n::tr("Authority")).color(t.text_label));
        ui.vertical(|ui| {
            let r =
                crate::text_field::TextField::singleline("field:ver.authority", &mut form.authority).hint("org.verstandard").width(300.0).show(ui);
            apply |= r.ending == Some(crate::text_field::Ending::Return);
            hint(ui, t, "The reverse-DNS domain you control; Records mint space ids under it.");
        });
        ui.end_row();
        ui.label(RichText::new(crate::i18n::tr("Exports")).color(t.text_label));
        if check(ui, "ver.exportSidecars", &mut sidecars, "Write VER Records beside exports") {
            configure(app, &mut form, json!({"exportSidecars": sidecars}));
        }
        ui.end_row();
        ui.label(RichText::new(crate::i18n::tr("Signing key")).color(t.text_label));
        ui.horizontal_wrapped(|ui| {
            let text = match (signing["keyId"].as_str(), signing["path"].as_str()) {
                (Some(key), _) => key.to_string(),
                (None, Some(path)) => path.to_string(),
                _ => crate::i18n::tr("None: Records are unsigned").to_string(),
            };
            let r = ui.label(RichText::new(text).font(t.font(11.5)).color(t.text));
            register(ui.ctx(), "ver:keyId", r.rect);
            if let Some(path) = signing["path"].as_str() {
                r.on_hover_text(path);
            }
            let r = ui.button(crate::i18n::tr("Choose…"));
            register(ui.ctx(), "button:verKeyChoose", r.rect);
            if r.clicked()
                && let Err(e) = app.run("dialog.verSigningKey", json!({}))
            {
                form.error = Some(super::faces::sentence(super::vera_settings::engine_message(&e)));
            }
            if signing["enabled"] == true {
                let r = ui.button(crate::i18n::tr("Clear"));
                register(ui.ctx(), "button:verKeyClear", r.rect);
                if r.clicked() {
                    configure(app, &mut form, json!({"signKeyPath": null}));
                }
            }
        });
        ui.end_row();
        ui.label(RichText::new(crate::i18n::tr("Default schema")).color(t.text_label));
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for (i, (value, label)) in SCHEMAS.iter().enumerate() {
                    if crate::widgets::text_button(ui, &format!("verSchema-{i}"), label, schema == *value).clicked() && schema != *value {
                        configure(app, &mut form, json!({"defaultSchema": value}));
                    }
                }
            });
            hint(ui, t, "VER 1.1 is a draft; 1.0.1 Records reach VER 1.0 readers. Exports and Write VER Records write both.");
        });
        ui.end_row();
    });
    if let Some(e) = signing["error"].as_str() {
        ui.add(egui::Label::new(RichText::new(super::faces::sentence(e)).font(t.font(11.5)).color(t.caution)).wrap());
    }
    if let Some(e) = &form.error {
        let r = ui.add(egui::Label::new(RichText::new(e).font(t.font(11.5)).color(t.caution)).wrap());
        register(ui.ctx(), "ver:error", r.rect);
    }
    let r = ui.button(crate::i18n::tr("Apply"));
    register(ui.ctx(), "button:verApply", r.rect);
    apply |= r.clicked();
    if apply {
        let authority = form.authority.trim();
        let value = if authority.is_empty() { Value::Null } else { json!(authority) };
        configure(app, &mut form, json!({"authority": value}));
    }
    hint(
        ui,
        t,
        "A VER Record says what a picture is (its fingerprints over the VER Canonical Buffer), what metadata it carried, where it came from and where its embedding lives. Photo → Show VER Record… shows one; File → Write VER Records… writes them for the selection.",
    );
    ctx.data_mut(|d| d.insert_temp(form_id(), form));
}

/// Settings ▸ Search ▸ VER Records ▸ Signing key ▸ Choose… once its dialog has answered `path`: the key is checked by
/// `ver.config` before it is kept.
pub fn set_signing_key(app: &mut LightcraftApp, ctx: &egui::Context, path: &str) -> Result<Value, String> {
    let r = app
        .session
        .execute("ver.config", &json!({"signKeyPath": path}))
        .map_err(|e| super::faces::sentence(super::vera_settings::engine_message(&e.to_string())));
    app.caches.vera_epoch += 1;
    match &r {
        Ok(_) => app.toast(ctx, crate::i18n::tr("VER Records are signed with this key from now on")),
        Err(e) => app.toast_error(ctx, e.clone()),
    }
    r
}

// ------------------------------------------------------------------------------------------ exports

/// The export dialog's "Write VER Records", for one run: the engine takes it from `ver.config {exportSidecars}` when
/// the export is set up, so it is switched to `on` for the run and put back after ([`restore_sidecars`]). `None` when
/// nothing needed changing; else what to put back.
pub fn sidecars_for_run(app: &mut LightcraftApp, on: bool) -> Option<bool> {
    let before = app.session.execute("ver.config", &json!({})).ok()?["exportSidecars"].as_bool()?;
    if before == on {
        return None;
    }
    app.session.execute("ver.config", &json!({"exportSidecars": on})).ok()?;
    Some(before)
}

/// Put Settings' "Write VER Records beside exports" back after an export that chose otherwise.
pub fn restore_sidecars(app: &mut LightcraftApp, before: bool) {
    // on is the default: put back as "the default", so the library's prefs read as they did
    let value = if before { Value::Null } else { json!(false) };
    if let Err(e) = app.session.execute("ver.config", &json!({"exportSidecars": value})) {
        log::warn!("ver.config: could not put the export sidecars setting back: {e}");
    }
}

/// Whether exports write Records now (the dialog's checkbox starts from it).
pub fn sidecars_on(app: &mut LightcraftApp) -> bool {
    app.session.execute("ver.config", &json!({})).ok().and_then(|v| v["exportSidecars"].as_bool()).unwrap_or(true)
}

/// The line an export's toast gets when some of its files have no Record (their rows' `notes`): once per batch, "VER: 3
/// of 12 files got no Record: needs the model".
pub fn export_notes(files: &[Value]) -> Option<String> {
    let written: Vec<&Value> = files.iter().filter(|f| f.get("path").is_some()).collect();
    let noted: Vec<&str> =
        written.iter().filter_map(|f| f["notes"].as_array()?.iter().filter_map(Value::as_str).find(|n| n.contains("VER Record"))).collect();
    let first = noted.first()?;
    let reason = if first.contains("search-by-meaning model") {
        crate::i18n::tr("needs the model").to_string()
    } else if first.contains("CPNP-1 path") {
        crate::i18n::tr("this format has no VER Canonical Buffer (a raw or DNG)").to_string()
    } else {
        first.trim_start_matches("No VER Record written: ").trim_start_matches("VER Record not written: ").trim_end_matches('.').to_string()
    };
    Some(crate::i18n::tr_format!("VER: {n} of {total} files got no Record: {reason}", n = noted.len(), total = written.len(), reason = reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_and_reasons_read_in_the_apps_words() {
        assert_eq!(
            error_code("invalid parameters for `ver.record`: a VER Record needs its canonical visual embedding: x (needs_model)"),
            "needs_model"
        );
        assert_eq!(error_code("something else"), "invalid");
        assert_eq!(engine_sentence("invalid parameters for `ver.record`: no VER Record: raw (no_cpnp_path)"), "no VER Record: raw");
        assert_eq!(reason_text("no_cpnp_path", ""), "This is a camera raw. It has no Record of its own; its exports do.");
        assert!(reason_text("needs_model", "").contains("search-by-meaning model"));
        assert_eq!(match_group("pixel_hash"), "Same pixels (a grouping, not identity)");
        assert_eq!(match_group("content"), "Same file");
        assert_eq!(match_group("pdq"), "Looks alike");
    }

    #[test]
    fn an_exports_ver_notes_are_counted_once() {
        let files = [
            json!({"path": "a.jpg", "notes": ["No VER Record written: it needs the search-by-meaning model for its canonical visual embedding. Download it."]}),
            json!({"path": "b.jpg"}),
            json!({"path": "c.jpg", "notes": ["No VER Record written: it needs the search-by-meaning model …"]}),
            json!({"skipped": "d.jpg"}),
        ];
        assert_eq!(export_notes(&files).as_deref(), Some("VER: 2 of 3 files got no Record: needs the model"));
        assert_eq!(export_notes(&files[1..2]), None);
    }

    #[test]
    fn the_status_line_says_what_records_lack() {
        let ready = json!({"ready": true, "signing": {"enabled": true, "keyId": "k1"}, "model": {"ready": true}});
        assert_eq!(status_line(&ready), ("Ready · signed as k1".to_string(), true, false));
        let unsigned = json!({"ready": true, "signing": {"enabled": false}, "model": {"ready": true}});
        assert_eq!(status_line(&unsigned).0, "Ready · unsigned");
        let no_model = json!({"ready": false, "signing": {"enabled": false}, "model": {"ready": false, "code": "needs_model"}, "reasons": ["x"]});
        assert_eq!(status_line(&no_model), ("Needs the model: Records carry an embedding".to_string(), false, true));
        let bad_key = json!({"ready": false, "signing": {"enabled": true, "error": "no such file"}, "model": {"ready": true}});
        assert_eq!(status_line(&bad_key).0, "The signing key can't be used: no such file");
    }
}
