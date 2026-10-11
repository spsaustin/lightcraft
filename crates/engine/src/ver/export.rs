//! Records written to files: the export hook (a Record beside every exported image) and `ver.export` (Records of
//! library photos written to a folder, now or in the background at the app's pace).
//!
//! **The export hook.** When an export writes an image and `ver.config {exportSidecars}` is on (the default), the
//! exported file gets its own Records beside it: `<name>.ver.json` (VER 1.1) and `<name>.ver-1.0.json` (VER 1.0, the
//! dual publication of VER 1.1 §1.1, sharing `content_hash`). They are made from the **exported bytes**: canonicalised,
//! hashed and embedded on the export's own thread right after encoding, then assembled once the file's final path is
//! known and written after it. The chain names what LightCraft did (`color_adjust`, `crop`, `resize`, `denoise`, or
//! `transcode` when it did none of those, then `export`) and, in 1.1, `lineage` names the source photo by its
//! `content_hash` and `pixel_hash` (`derived`, or `exported` when the rendition's pixels are the source's). A raw
//! source has no Record to name, so its content hash is carried in the `extensions` block instead. Nothing here ever
//! fails an export: when no Record can be made (no model, no CPNP-1 path for the format) the export's result says so in
//! one line of `notes`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lightcraft_catalog::{PhotoId, Source};
use serde_json::{Value, json};

use super::source::EmbeddingSource;
use super::{Analysis, Context, Event, Facts, GENERATOR, Level, Schema, VerError, util};
use crate::Session;

/// The sidecar extensions of an export: the 1.1 Record, then the 1.0 one.
pub(crate) const SIDECAR_1_1: &str = "ver.json";
pub(crate) const SIDECAR_1_0: &str = "ver-1.0.json";

/// The source photo of an export, as its Record's lineage needs it.
#[derive(Clone)]
struct Parent {
    path: Option<String>,
    read: Option<crate::merge::ByteReader>,
    /// The digest lane's fingerprint of the source, when it has one (saves a full-size decode).
    cached: Option<crate::vera::digests::Fingerprint>,
}

/// What an export's Records need from the session, taken when the export is set up.
#[derive(Clone)]
pub struct Hook {
    ctx: Context,
    source: Arc<dyn EmbeddingSource>,
    parent: Parent,
    /// The develop events the rendition went through.
    edits: Vec<&'static str>,
    /// Whether the Records may name the files' locations (not when the export strips metadata).
    keep_paths: bool,
    started_at: String,
}

/// The heavy half, done on the export's thread: the exported bytes analysed, the source identified.
pub struct Pending {
    hook: Hook,
    analysis: Result<Analysis, VerError>,
    /// The source's (content_hash, pixel_hash), `pixel_hash` `None` without a CPNP-1 path (`raw` says why).
    parent: Option<(String, Option<String>, bool)>,
    exported_at: String,
}

/// The export hook for photo `id` under `o`, or `None` when its Records are off (`ver.config {exportSidecars:
/// false}`) or the export writes no picture of its own (a contact sheet).
pub(crate) fn prepare(s: &mut Session, id: PhotoId, o: &crate::export::ExportOptions) -> Option<Hook> {
    if !s.ver.prefs.export_sidecars() {
        return None;
    }
    let ctx = match super::context(s) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("ver: no Records for this export: {e}");
            return None;
        }
    };
    let source = super::embedding_source(s);
    crate::vera::ensure_open(s);
    let p = s.catalog.photo(id)?.clone();
    let key = crate::media::content_key(&p);
    let path = match &p.source {
        Source::File { path } => Some(path.clone()),
        Source::Demo { .. } => None,
    };
    let mut edits = Vec::new();
    let rendered = o.format.is_rendered();
    if rendered && p.is_edited() {
        edits.push("color_adjust");
    }
    if rendered && p.is_cropped() {
        edits.push("crop");
    }
    if rendered && s.media.denoise.get(id).is_some() {
        edits.push("denoise");
    }
    if rendered {
        let (w, h) = crate::export::output_size(&p, o);
        let native = lightcraft_pipeline::native_output_size(p.width.max(1) as usize, p.height.max(1) as usize, &p.develop);
        if (w, h) != ((native.0.round() as usize).max(1), (native.1.round() as usize).max(1)) {
            edits.push("resize");
        }
    }
    Some(Hook {
        ctx,
        source,
        parent: Parent { path, read: s.media.file_bytes.clone(), cached: s.vera.digests.fingerprint(&key).copied() },
        edits,
        keep_paths: o.metadata != crate::export::MetadataPolicy::None,
        started_at: util::now(),
    })
}

impl Hook {
    /// Analyse the exported `bytes` and identify the source (on the export's thread).
    pub fn analyze(self, bytes: &[u8]) -> Pending {
        // the rendition exists once its bytes do: the export event precedes the decode that analyses them (a chain
        // runs forwards, VER1201)
        let exported_at = util::now();
        let analysis = super::analyze(bytes, self.source.as_ref());
        let parent = if analysis.is_ok() { self.identify_parent() } else { None };
        Pending { hook: self, analysis, parent, exported_at }
    }

    fn identify_parent(&self) -> Option<(String, Option<String>, bool)> {
        let path = self.parent.path.as_ref()?;
        let bytes = match &self.parent.read {
            Some(r) => r(path),
            None => std::fs::read(path).map_err(|e| e.to_string()),
        }
        .ok()?;
        let content = util::sha256_hex(&bytes);
        let raw = lightcraft_raw::probe(&bytes).is_some();
        let pixel = match self.parent.cached.and_then(|f| f.pixel.zip(f.content)) {
            Some((pixel, c)) if util::hex(&c) == content => Some(util::hex(&pixel)),
            _ if raw => None,
            _ => crate::vera::fingerprint::canonical(&bytes).ok().and_then(|f| f.pixel).map(|p| util::hex(&p)),
        };
        Some((content, pixel, raw))
    }
}

/// The sidecar files of an exported file at `main`.
pub(crate) fn sidecar_names(main: &str) -> [String; 2] {
    let p = Path::new(main);
    [p.with_extension(SIDECAR_1_1).to_string_lossy().to_string(), p.with_extension(SIDECAR_1_0).to_string_lossy().to_string()]
}

/// One line for an export's `notes` saying why no Record was written.
fn note(e: &VerError) -> String {
    match e {
        VerError::NeedsModel(m) => format!("No VER Record written: it needs the search-by-meaning model for its canonical visual embedding. {m}"),
        VerError::NoCpnpPath(_) => {
            "No VER Record written: this file format has no CPNP-1 path (a raw, DNG, JPEG XL, …), so it has no Canonical Buffer.".into()
        }
        e => format!("No VER Record written: {e}"),
    }
}

impl Pending {
    /// The Records of the file written at `main`, as `(path, bytes)` (the 1.1 one first), or the note saying why there
    /// are none.
    pub fn finish(&self, main: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
        let a = self.analysis.as_ref().map_err(note)?;
        let h = &self.hook;
        let main_uri = util::file_uri(Path::new(main));
        let mut events: Vec<Event> = h.edits.iter().map(|e| Event { action: e, at: h.started_at.clone(), extra_1_1: json!({}) }).collect();
        let raw_parent = self.parent.as_ref().is_some_and(|p| p.2);
        if raw_parent && !h.edits.contains(&"color_adjust") {
            // a raw is always developed into a rendition
            events.insert(0, Event { action: "color_adjust", at: h.started_at.clone(), extra_1_1: json!({}) });
        }
        if events.is_empty() {
            events.push(Event { action: "transcode", at: h.started_at.clone(), extra_1_1: json!({}) });
        }
        let export_extra = if h.keep_paths { json!({"uri": main_uri}) } else { json!({}) };
        events.push(Event { action: "export", at: self.exported_at.clone(), extra_1_1: export_extra });
        let (lineage, source_note) = match &self.parent {
            Some((content, _, _)) if *content == a.identity.content_hash => (None, None),
            Some((content, Some(pixel), false)) => {
                let relation = if *pixel == a.identity.pixel_hash { "exported" } else { "derived" };
                let domain = super::space::domain(&h.ctx.authority);
                let parent = json!({
                    "producer_id": h.ctx.producer_id,
                    "record_id": util::record_id(&domain, &h.ctx.producer_id, content, Schema::V1_1Draft.ver_version()),
                    "content_hash": content,
                    "pixel_hash": {"value": pixel, "cpnp_version": "1.0"},
                    "relation": relation,
                });
                (Some(parent), None)
            }
            Some((content, _, raw)) => {
                let why = if *raw { "a camera raw: no CPNP-1 path, so no Record of its own" } else { "no CPNP-1 path, so no Record of its own" };
                (None, Some(json!({"content_hash": content, "relation": "derived", "note": why})))
            }
            None => (None, None),
        };
        let facts = Facts {
            uri: h.keep_paths.then(|| main_uri.clone()),
            acquisition: Some(("other".into(), h.started_at.clone(), json!({"actor": GENERATOR}))),
            events,
            platform: None,
            lineage,
            source_note,
        };
        let (r10, r11) = super::produce_both(&h.ctx, a, &facts, Level::L3).map_err(|e| note(&e))?;
        let [p11, p10] = sidecar_names(main);
        Ok(vec![(p11, r11.record.to_json_pretty().into_bytes()), (p10, r10.record.to_json_pretty().into_bytes())])
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// `ver.export`: Records of library photos written to a folder

/// One photo's Records to write.
pub(crate) struct Job {
    pub photo: PhotoId,
    pub key: String,
    pub path: String,
    read: Option<crate::merge::ByteReader>,
    facts: Facts,
    ctx: Context,
    source: Arc<dyn EmbeddingSource>,
    schemas: Vec<Schema>,
    dir: PathBuf,
    /// The library's originals, never written over.
    guard: Arc<crate::originals::OriginalGuard>,
}

/// What one job did.
pub(crate) struct Done {
    pub photo: PhotoId,
    pub key: String,
    pub result: Result<(Vec<String>, Analysis, super::Produced), VerError>,
}

/// The schemas a `ver.export` `schema` parameter names: `both` (the default), `1.1-draft` or `1.0.1`.
pub(crate) fn schemas_param(v: Option<&str>) -> Option<Vec<Schema>> {
    match v {
        None | Some("both") => Some(vec![Schema::V1_1Draft, Schema::V1_0_1]),
        Some(s) => super::parse_schema(s).map(|s| vec![s]),
    }
}

/// A job for photo `id` (cheap: the file is read by [`run`]).
pub(crate) fn job(
    s: &mut Session,
    id: PhotoId,
    dir: &Path,
    schemas: &[Schema],
    guard: &Arc<crate::originals::OriginalGuard>,
) -> Result<Job, VerError> {
    let p = s.catalog.photo(id).cloned().ok_or_else(|| VerError::Invalid(format!("no photo {}", id.0)))?;
    let Source::File { path } = &p.source else {
        return Err(VerError::NoAsset(format!("{} is a generated demo photo: it has no file, so it has no Record", p.file_name)));
    };
    let facts = super::photo_facts(s, &p, path);
    Ok(Job {
        photo: id,
        key: crate::media::content_key(&p),
        path: path.clone(),
        read: s.media.file_bytes.clone(),
        facts,
        ctx: super::context(s)?,
        source: super::embedding_source(s),
        schemas: schemas.to_vec(),
        dir: dir.to_path_buf(),
        guard: guard.clone(),
    })
}

/// Make and write one job's Records (any thread): `<content_hash>.ver.json` for the first schema, and
/// `<content_hash>.ver-1.0.json` for a 1.0 twin.
pub(crate) fn run(job: Job, write: &dyn Fn(&str, &[u8]) -> Result<(), String>) -> Done {
    let result = (|| {
        let bytes = match &job.read {
            Some(r) => r(&job.path),
            None => std::fs::read(&job.path).map_err(|e| format!("{}: {e}", job.path)),
        }
        .map_err(VerError::Source)?;
        let a = super::analyze(&bytes, job.source.as_ref())?;
        drop(bytes);
        let mut files = Vec::new();
        let mut first = None;
        for (i, schema) in job.schemas.iter().enumerate() {
            let r = super::assemble(&job.ctx, &a, &job.facts, *schema, Level::L3)?;
            let ext = if i > 0 && *schema == Schema::V1_0_1 { SIDECAR_1_0 } else { SIDECAR_1_1 };
            let path = job.dir.join(format!("{}.{ext}", a.identity.content_hash)).to_string_lossy().to_string();
            job.guard.check(Path::new(&path)).map_err(VerError::Source)?;
            write(&path, r.record.to_json_pretty().as_bytes()).map_err(VerError::Source)?;
            files.push(path);
            first.get_or_insert(r);
        }
        let first = first.ok_or_else(|| VerError::Invalid("no schema".into()))?;
        Ok((files, a, first))
    })();
    Done { photo: job.photo, key: job.key, result }
}

/// Background `ver.export` work, done at the app's pace by `vera.pump`.
#[derive(Default)]
pub(crate) struct Queue {
    jobs: Vec<Job>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<Worker>,
    in_flight: usize,
    written: usize,
    failed: Vec<Value>,
    peak: usize,
    task: Option<crate::activity::TaskGuard>,
}

impl Queue {
    pub fn push(&mut self, job: Job) {
        self.jobs.push(job);
    }
    pub fn pending(&self) -> usize {
        self.jobs.len() + self.in_flight
    }
}

/// The thread that writes Records in the background.
#[cfg(not(target_arch = "wasm32"))]
struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    done: std::sync::mpsc::Receiver<Done>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Worker {
    fn start() -> Option<Worker> {
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let (tx, done) = std::sync::mpsc::channel::<Done>();
        std::thread::Builder::new()
            .name("lightcraft-ver".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let (photo, key) = (job.photo, job.key.clone());
                    let d = crate::guard::catch("writing a VER Record", || run(job, &|p, b| crate::export::write_file(p, b)))
                        .unwrap_or_else(|e| Done { photo, key, result: Err(VerError::Invalid(e)) });
                    if tx.send(d).is_err() {
                        return;
                    }
                }
            })
            .ok()
            .map(|_| Worker { jobs, done })
    }
}

/// Take in what a finished job learned: the photo's fingerprint for the digest lane, its vector for the consumer.
pub(crate) fn absorb(s: &mut Session, d: &Done) {
    if let Ok((_, a, p)) = &d.result {
        super::remember(s, &d.key, a);
        super::note_produced(s, p, Some(d.photo), p.record.schema().unwrap_or(Schema::V1_1Draft));
    }
}

/// The background part of one `vera.pump` call: `{pending, written, failed}`.
pub(crate) fn pump(s: &mut Session, pace: crate::vera::Pace) -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let finished: Vec<Done> = s.ver.queue.worker.as_ref().map(|w| w.done.try_iter().collect()).unwrap_or_default();
        for d in finished {
            s.ver.queue.in_flight = s.ver.queue.in_flight.saturating_sub(1);
            absorb(s, &d);
            match &d.result {
                Ok(_) => s.ver.queue.written += 1,
                Err(e) => s.ver.queue.failed.push(json!({"id": d.photo.0, "code": e.code(), "reason": e.to_string()})),
            }
        }
        // one Record in flight at a time, none while paused
        if pace.per_call() > 0 && s.ver.queue.in_flight == 0 && !s.ver.queue.jobs.is_empty() {
            if s.ver.queue.worker.is_none() {
                s.ver.queue.worker = Worker::start();
            }
            let job = s.ver.queue.jobs.remove(0);
            match &s.ver.queue.worker {
                Some(w) => match w.jobs.send(job) {
                    Ok(()) => s.ver.queue.in_flight += 1,
                    Err(e) => {
                        s.ver.queue.worker = None;
                        s.ver.queue.jobs.insert(0, e.0);
                    }
                },
                None => s.ver.queue.jobs.insert(0, job),
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        // no threads and no file system to write to: the queue is refused by `ver.export` on the web
        let _ = pace;
    }
    let q = &mut s.ver.queue;
    let pending = q.pending();
    if pending == 0 {
        q.task = None;
        q.peak = 0;
    } else {
        q.peak = q.peak.max(pending);
        let (done, total) = (q.peak.saturating_sub(pending) as u64, q.peak as u64);
        let task = q.task.get_or_insert_with(|| s.activity.start("ver", "Writing VER records", crate::activity::Cancel::No));
        task.progress(done, total);
    }
    let q = &s.ver.queue;
    json!({"pending": pending, "written": q.written, "failed": q.failed})
}
