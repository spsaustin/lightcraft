//! The Describe lane's remote path, native only (`docs/vera.md` → Phase 3): an OpenAI-compatible
//! `POST …/v1/chat/completions` the user points LightCraft at (llama.cpp, Ollama, vLLM, LM Studio on their own machine,
//! or a host they trust), with each photo as a fresh 1024 px JPEG.
//!
//! - **Policy** ([`endpoint`]): plain `http://` only to this computer (127.0.0.0/8, `::1`, `localhost`); any other host
//!   needs `https://` *and* the user's consent (`allowRemoteHost: true`), because the photos leave the machine.
//! - **The token** ([`read_token`], [`write_token`]): `describe-remote.token` beside the model folders, created mode
//!   0600 and replaced atomically, refused when group- or world-readable (as ssh does); `LIGHTCRAFT_DESCRIBE_TOKEN`
//!   overrides it. It lives in memory only for the requests of a run and is never in a command's output, a log line,
//!   an error message, the journal or the catalog.
//! - **The worker** ([`Pool`]): [`THREADS`] threads, so at most that many requests are in flight. Each takes one photo
//!   end to end: render (the face scan's path, default settings, upright, uncropped, at [`IMAGE_EDGE`]), JPEG at
//!   quality [`JPEG_QUALITY`] (no metadata: nothing but pixels leaves the machine), the request, its retries, and the
//!   zero-shot keywords. [`ATTEMPTS`] attempts a photo; a `429` waits out `Retry-After` for every thread (one 429
//!   pauses all requests) and does not spend an attempt; a `400` steps the response format down (strict JSON schema →
//!   JSON mode → none, remembered for the endpoint) without spending one; `401` / `403` end the photo at once; any
//!   other failure spends one. [`STALL`] without data is a stall; a response over [`MAX_RESPONSE`] is refused.
//!
//! The session thread never waits for any of it: it hands jobs over and takes results in (`vera.pump`).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use lightcraft_catalog::PhotoId;
use lightcraft_fetch::http::{self, HttpError, Limits, Url};
use serde_json::{Value, json};
use web_time::Instant;

use super::keywords::{Calibration, Matrix};
use super::vocabulary::Vocabulary;
use crate::faces_worker::{Pace as ScanPace, scan_pool};
use crate::media::{PreviewLoader, RenderJob};

/// The token file, beside the model folders (`<config>/models`).
pub(crate) const TOKEN_FILE: &str = "describe-remote.token";
/// The environment variable that overrides the token file.
pub(crate) const TOKEN_ENV: &str = "LIGHTCRAFT_DESCRIBE_TOKEN";
/// Longest token kept.
const MAX_TOKEN: usize = 4096;
/// Longest endpoint URL and model name.
pub(crate) const MAX_ENDPOINT: usize = 2048;
pub(crate) const MAX_MODEL_NAME: usize = 256;
/// Requests in flight at once (one per worker thread).
pub(crate) const THREADS: usize = 4;
/// Attempts per photo (a 429 or a format step-down does not count).
pub(crate) const ATTEMPTS: u32 = 3;
/// Most 429s one photo waits out before it fails.
const MAX_RATE_LIMITED: u32 = 30;
/// The longest `Retry-After` honoured, and the wait when a 429 names none.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(300);
const DEFAULT_RETRY_AFTER: Duration = Duration::from_secs(5);
/// Connecting, and waiting for data once connected.
const CONNECT: Duration = Duration::from_secs(15);
pub(crate) const STALL: Duration = Duration::from_secs(120);
/// Largest response read.
pub(crate) const MAX_RESPONSE: usize = 64 * 1024;
/// The picture sent: long edge and JPEG quality.
pub(crate) const IMAGE_EDGE: usize = 1024;
pub(crate) const JPEG_QUALITY: u8 = 85;
/// Between failed attempts.
const BACKOFF: Duration = Duration::from_millis(250);
/// Field limits of a description (characters; keywords: count and characters each).
pub(crate) const MAX_TITLE: usize = 80;
pub(crate) const MAX_CAPTION: usize = 300;
pub(crate) const MAX_ALT: usize = 300;
pub(crate) const MAX_REMOTE_KEYWORDS: usize = 20;
pub(crate) const MAX_KEYWORD: usize = 40;

const SYSTEM: &str = "You write catalog metadata for a photographer's archive. Describe only what is visible in the photograph. Do not guess who a person is, or their gender, age, ethnicity, religion or other personal attributes. Answer with JSON only.";
const USER: &str = "Describe this photograph for the catalog. Return one JSON object with these keys: \"title\": at most 8 words in Title Case, without a final period; \"caption\": one sentence; \"alt\": alt text of 1 to 2 sentences, at most 250 characters; \"keywords\": 5 to 15 lower-case keywords. JSON only, no other text.";

/// Why an endpoint cannot be used.
pub(crate) fn endpoint(url: &str, allow_remote_host: bool) -> Result<Url, String> {
    if url.len() > MAX_ENDPOINT {
        return Err(format!("the endpoint URL is longer than {MAX_ENDPOINT} characters"));
    }
    let u = Url::parse(url).map_err(|e| format!("the endpoint is not a usable URL ({e})"))?;
    if u.path.contains('?') {
        return Err("the endpoint URL may not carry a query (put the token in vera.describe.config's `token`)".into());
    }
    if u.is_loopback() {
        return Ok(u);
    }
    if !u.tls {
        return Err(format!(
            "plain http:// is only allowed to this computer (127.0.0.1, ::1, localhost); use https:// for {} (or an SSH tunnel to localhost)",
            u.host
        ));
    }
    if !allow_remote_host {
        return Err(format!(
            "{} is another computer: describing photos there sends their pixels to it. Pass `allowRemoteHost: true` once the user has agreed",
            u.host
        ));
    }
    Ok(u)
}

/// The chat-completions URL of an endpoint: as given when it already ends in `/chat/completions`, `…/chat/completions`
/// after a `/v1`, else `…/v1/chat/completions`.
pub(crate) fn completions_url(base: &Url) -> Url {
    let path = base.path.trim_end_matches('/');
    let path = if path.ends_with("/chat/completions") {
        path.to_string()
    } else if path.ends_with("/v1") {
        format!("{path}/chat/completions")
    } else {
        format!("{path}/v1/chat/completions")
    };
    Url { path, ..base.clone() }
}

/// The host as `vera.status` and the activity row show it (`name[:port]`).
pub(crate) fn host_of(u: &Url) -> String {
    let host = if u.host.contains(':') { format!("[{}]", u.host) } else { u.host.clone() };
    let default = if u.tls { 443 } else { 80 };
    if u.port == default { host } else { format!("{host}:{}", u.port) }
}

/// Where the token is kept: beside the model folder (`<vera_model_dir>/../describe-remote.token`).
pub(crate) fn token_path(model_dir: Option<&Path>) -> Option<PathBuf> {
    Some(model_dir?.parent()?.join(TOKEN_FILE))
}

/// A token as it may be sent: 1–4096 visible ASCII characters (no spaces, no line breaks: it goes in a header).
pub(crate) fn valid_token(t: &str) -> Result<(), String> {
    if t.is_empty() || t.len() > MAX_TOKEN || !t.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(format!("a token is 1 to {MAX_TOKEN} visible ASCII characters, without spaces"));
    }
    Ok(())
}

/// The token to send: `env` (the value of [`TOKEN_ENV`]) when set, else the file at `path`. `Ok(None)` when there is
/// none; an error for a file other users can read, or one that is not a token.
pub(crate) fn read_token(path: Option<&Path>, env: Option<String>) -> Result<Option<String>, String> {
    if let Some(t) = env.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
        valid_token(&t).map_err(|e| format!("{TOKEN_ENV}: {e}"))?;
        return Ok(Some(t));
    }
    let Some(path) = path else { return Ok(None) };
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("the describe token file cannot be read: {e}")),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o077 != 0 {
            return Err(format!(
                "the describe token file {} can be read by other users; it is not used until only you can (chmod 600), or set it again with vera.describe.config",
                path.display()
            ));
        }
    }
    if !meta.is_file() || meta.len() > MAX_TOKEN as u64 + 2 {
        return Err("the describe token file is not a token".into());
    }
    let bytes = crate::denoise::read_capped(path, MAX_TOKEN + 2).ok_or("the describe token file cannot be read")?;
    let t = String::from_utf8(bytes).map_err(|_| "the describe token file is not a token".to_string())?;
    let t = t.trim();
    if t.is_empty() {
        return Ok(None);
    }
    valid_token(t).map_err(|_| "the describe token file is not a token".to_string())?;
    Ok(Some(t.to_string()))
}

/// The token from the environment, if set.
pub(crate) fn token_env() -> Option<String> {
    std::env::var(TOKEN_ENV).ok()
}

/// Keep `token` at `path` (mode 0600 from the start, replaced atomically), or delete the file for `None`.
pub(crate) fn write_token(path: &Path, token: Option<&str>) -> Result<(), String> {
    let Some(token) = token else {
        return match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("could not delete the describe token: {e}")),
        };
    };
    valid_token(token)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let part = path.with_extension("token.part");
    let _ = std::fs::remove_file(&part);
    let written = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&part)?;
        f.write_all(token.as_bytes())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&part, path)
    })();
    written.map_err(|e| {
        let _ = std::fs::remove_file(&part);
        format!("could not save the describe token: {e}")
    })
}

/// Where a run sends its requests. Not `Debug`: it holds the token.
pub(crate) struct Remote {
    pub url: Url,
    pub host: String,
    pub model: String,
    token: Option<String>,
}

impl Remote {
    pub fn new(base: &Url, model: &str, token: Option<String>) -> Remote {
        Remote { url: completions_url(base), host: host_of(base), model: model.to_string(), token }
    }

    /// `text` with the token taken out (a server's error message could echo a header).
    fn scrub(&self, text: &str) -> String {
        match &self.token {
            Some(t) if !t.is_empty() => text.replace(t.as_str(), "…"),
            _ => text.to_string(),
        }
    }
}

/// What the endpoint is asked to answer in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Schema = 0,
    JsonObject = 1,
    None = 2,
}

impl Format {
    fn from_u8(v: u8) -> Format {
        match v {
            0 => Format::Schema,
            1 => Format::JsonObject,
            _ => Format::None,
        }
    }
}

/// The request body for one photo.
pub(crate) fn request_body(model: &str, jpeg_base64: &str, format: Format) -> Value {
    let mut body = json!({
        "model": model,
        "temperature": 0.2,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": USER},
                {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{jpeg_base64}")}},
            ]},
        ],
    });
    match format {
        Format::Schema => {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "photo_description",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "maxLength": MAX_TITLE},
                            "caption": {"type": "string", "maxLength": MAX_CAPTION},
                            "alt": {"type": "string", "maxLength": MAX_ALT},
                            "keywords": {"type": "array", "items": {"type": "string", "maxLength": MAX_KEYWORD}, "maxItems": MAX_REMOTE_KEYWORDS},
                        },
                        "required": ["title", "caption", "alt", "keywords"],
                        "additionalProperties": false,
                    },
                },
            });
        }
        Format::JsonObject => body["response_format"] = json!({"type": "json_object"}),
        Format::None => {}
    }
    body
}

/// A description as the endpoint wrote it, checked and trimmed to the limits.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Described {
    pub title: String,
    pub caption: String,
    pub alt_text: String,
    pub keywords: Vec<String>,
}

/// `s` cut to `max` characters (at a character boundary), control characters turned into spaces, trimmed.
fn clip(s: &str, max: usize) -> String {
    let clean: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let words = clean.split_whitespace().collect::<Vec<_>>().join(" ");
    words.chars().take(max).collect::<String>().trim().to_string()
}

/// The text inside a Markdown code fence, if `s` is one.
fn strip_fences(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else { return t };
    // the fence's info string (`json`) runs to the end of its line
    let rest = rest.split_once('\n').map_or("", |(_, r)| r);
    rest.trim_end().strip_suffix("```").unwrap_or(rest).trim()
}

/// The first balanced `{…}` in `s` (strings and escapes respected), for a model that wraps its JSON in prose.
fn first_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let (mut depth, mut in_str, mut esc) = (0usize, false, false);
    for (i, c) in s.get(start..)?.char_indices() {
        if in_str {
            match (esc, c) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return s.get(start..=start + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// The message the model wrote, from a chat-completions response body.
fn message_content(body: &[u8]) -> Result<String, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("the endpoint's answer is not JSON ({e})"))?;
    let content = v.pointer("/choices/0/message/content").ok_or("the endpoint's answer has no choices[0].message.content")?;
    match content {
        Value::String(t) => Ok(t.clone()),
        // a list of parts: their text, in order
        Value::Array(parts) => Ok(parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("")),
        _ => Err("the endpoint's message content is not text".into()),
    }
}

/// The description in a chat-completions response body: the message's JSON (code fences stripped; the first `{…}` when
/// there is prose around it), every field the right type, trimmed to its limit.
pub(crate) fn parse_response(body: &[u8]) -> Result<Described, String> {
    let content = message_content(body)?;
    let text = strip_fences(&content);
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => first_object(text).and_then(|o| serde_json::from_str(o).ok()).ok_or("the model's answer holds no JSON object")?,
    };
    let obj = v.as_object().ok_or("the model's answer is not a JSON object")?;
    let text_field = |names: &[&str], max: usize| -> Result<String, String> {
        for n in names {
            match obj.get(*n) {
                None | Some(Value::Null) => {}
                Some(Value::String(s)) => return Ok(clip(s, max)),
                Some(_) => return Err(format!("the model's `{n}` is not text")),
            }
        }
        Ok(String::new())
    };
    let mut title = text_field(&["title"], MAX_TITLE)?;
    while title.ends_with('.') {
        title.pop();
    }
    let caption = text_field(&["caption"], MAX_CAPTION)?;
    let alt_text = text_field(&["alt", "alt_text", "altText"], MAX_ALT)?;
    let mut keywords: Vec<String> = Vec::new();
    match obj.get("keywords") {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) => {
            for k in a {
                let k = k.as_str().ok_or("the model's keywords are not all text")?;
                // `|` makes a keyword hierarchy in LightCraft: not the model's to decide
                let k = clip(&k.replace('|', " "), MAX_KEYWORD).to_lowercase();
                if !k.is_empty() && !keywords.contains(&k) && keywords.len() < MAX_REMOTE_KEYWORDS {
                    keywords.push(k);
                }
            }
        }
        // a comma-separated string, as small models sometimes write
        Some(Value::String(s)) => {
            for k in s.split(',') {
                let k = clip(&k.replace('|', " "), MAX_KEYWORD).to_lowercase();
                if !k.is_empty() && !keywords.contains(&k) && keywords.len() < MAX_REMOTE_KEYWORDS {
                    keywords.push(k);
                }
            }
        }
        Some(_) => return Err("the model's `keywords` is not a list".into()),
    }
    if title.is_empty() && caption.is_empty() && alt_text.is_empty() && keywords.is_empty() {
        return Err("the model's answer describes nothing".into());
    }
    Ok(Described { title, caption, alt_text, keywords })
}

/// `Retry-After` as a wait (seconds only; a date or nothing is the default), capped.
pub(crate) fn retry_after(v: Option<&str>) -> Duration {
    v.and_then(|s| s.trim().parse::<u64>().ok()).map_or(DEFAULT_RETRY_AFTER, |n| Duration::from_secs(n).min(MAX_RETRY_AFTER))
}

/// Standard base64 with padding.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(*c.first().unwrap_or(&0)) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        let ch = |i: u32| char::from(*T.get(((n >> i) & 63) as usize).unwrap_or(&b'A'));
        out.push(ch(18));
        out.push(ch(12));
        out.push(if c.len() > 1 { ch(6) } else { '=' });
        out.push(if c.len() > 2 { ch(0) } else { '=' });
    }
    out
}

/// What the worker threads share: Pause (no new request is sent while it holds), the rate-limit pause one 429 sets for
/// all of them, and the response format the endpoint accepts.
#[derive(Default)]
pub(crate) struct Shared {
    pub paused: AtomicBool,
    until: Mutex<Option<Instant>>,
    format: AtomicU8,
}

impl Shared {
    /// A new endpoint or model: start again from the strict schema.
    pub fn reset_format(&self) {
        self.format.store(Format::Schema as u8, Ordering::Relaxed);
    }

    fn pause_for(&self, d: Duration) {
        let mut u = self.until.lock().unwrap_or_else(PoisonError::into_inner);
        let t = Instant::now() + d;
        if u.is_none_or(|x| x < t) {
            *u = Some(t);
        }
    }

    /// Wait until a request may be sent: not paused by the app, not rate-limited. `false` when cancelled meanwhile.
    fn wait_ready(&self, cancel: &AtomicBool) -> bool {
        loop {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            let wait = {
                let u = self.until.lock().unwrap_or_else(PoisonError::into_inner);
                u.and_then(|t| t.checked_duration_since(Instant::now()))
            };
            match wait {
                Some(w) if !w.is_zero() => std::thread::sleep(w.min(Duration::from_millis(50))),
                _ if self.paused.load(Ordering::Relaxed) => std::thread::sleep(Duration::from_millis(50)),
                _ => return true,
            }
        }
    }
}

/// The zero-shot keywords part of a job.
pub(crate) struct KeywordJob {
    pub vocab: Arc<Vocabulary>,
    pub matrix: Arc<Matrix>,
    pub calibration: Option<Arc<Calibration>>,
    /// The photo's canonical vector, when it has one; else the worker makes it with `model`.
    pub vector: Option<Vec<f32>>,
    pub model: Option<Arc<lightcraft_embed::Model>>,
    pub limit: usize,
}

/// One photo to describe.
pub(crate) struct Job {
    pub epoch: u64,
    pub photo: PhotoId,
    pub key: String,
    /// The 1024 px render for the endpoint (and for the photo's vector when it has none yet).
    pub render: Option<RenderJob>,
    /// For keywords alone, when the photo has no vector yet: the embed lane's render and embedded preview.
    pub embed_render: Option<(RenderJob, Option<(String, PreviewLoader)>)>,
    pub remote: Option<Arc<Remote>>,
    pub keywords: Option<KeywordJob>,
    pub cancel: Arc<AtomicBool>,
}

/// A finished job.
pub(crate) struct Done {
    pub epoch: u64,
    pub photo: PhotoId,
    pub key: String,
    pub result: Result<Outcome, String>,
    /// A canonical vector the worker made on the way (for the vector lane's store).
    pub vector: Option<Vec<f32>>,
}

/// What a job proposes.
pub(crate) struct Outcome {
    pub described: Option<Described>,
    pub local: Option<super::keywords::Scored>,
    pub model: Option<String>,
    pub host: Option<String>,
}

/// The worker threads and their channels. Dropping it ends the threads once they finish what they hold.
pub(crate) struct Pool {
    jobs: Sender<Job>,
    done: Receiver<Done>,
    pub shared: Arc<Shared>,
}

impl Pool {
    pub fn start() -> Option<Pool> {
        let (jobs, rx) = channel::<Job>();
        let (tx, done) = channel::<Done>();
        let rx = Arc::new(Mutex::new(rx));
        let shared = Arc::new(Shared::default());
        let mut started = 0;
        for i in 0..THREADS {
            let (rx, tx, shared) = (rx.clone(), tx.clone(), shared.clone());
            let spawned = std::thread::Builder::new().name(format!("lightcraft-vera-describe-{i}")).spawn(move || {
                loop {
                    // one thread waits for the next job at a time; the lock is let go before the work
                    let job = { rx.lock().unwrap_or_else(PoisonError::into_inner).recv() };
                    let Ok(job) = job else { return };
                    let (epoch, photo, key) = (job.epoch, job.photo, job.key.clone());
                    let run = catch_unwind(AssertUnwindSafe(|| run(job, &shared)));
                    let (result, vector) = run.unwrap_or_else(|_| (Err("describing stopped unexpectedly".into()), None));
                    if tx.send(Done { epoch, photo, key, result, vector }).is_err() {
                        return;
                    }
                }
            });
            if spawned.is_ok() {
                started += 1;
            }
        }
        (started > 0).then_some(Pool { jobs, done, shared })
    }

    pub fn submit(&self, job: Job) -> bool {
        self.jobs.send(job).is_ok()
    }

    /// Everything finished since the last call.
    pub fn finished(&self) -> Vec<Done> {
        self.done.try_iter().collect()
    }
}

/// Run `f` on the face scan's light pool (so a render's parallel parts stay off rayon's global pool).
fn on_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match scan_pool(ScanPace::Light) {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// One job, end to end: `(proposal or error, a canonical vector made on the way)`.
fn run(mut job: Job, shared: &Shared) -> (Result<Outcome, String>, Option<Vec<f32>>) {
    if job.cancel.load(Ordering::Relaxed) {
        return (Err(CANCELLED.into()), None);
    }
    // the picture for the endpoint
    let image = match job.render.take() {
        Some(r) => match on_pool(|| r.run().rendered) {
            Ok(r) => Some(r.image),
            Err(e) => return (Err(format!("the photo could not be rendered: {e}")), None),
        },
        None => None,
    };
    // the zero-shot keywords, and the photo's vector when it had none
    let mut made = None;
    let local = match job.keywords.take() {
        Some(k) => {
            let vector = match k.vector {
                Some(v) => Some(v),
                None => {
                    let pic = match (&image, job.embed_render.take()) {
                        (Some(img), _) => super::worker::squash(img, lightcraft_embed::INPUT).ok_or_else(|| "the picture is empty".to_string()),
                        (None, Some((render, preview))) => on_pool(|| super::worker::picture(render, preview)),
                        (None, None) => Err("nothing to embed".into()),
                    };
                    let v = match (pic, &k.model) {
                        (Ok(pic), Some(m)) => match catch_unwind(AssertUnwindSafe(|| on_pool(|| m.embed_image(&pic)))) {
                            Ok(Ok(v)) => Some(v),
                            Ok(Err(e)) => {
                                log::debug!("vera: describe: photo {} not embedded: {e}", job.photo.0);
                                None
                            }
                            Err(_) => None,
                        },
                        (Err(e), _) => {
                            log::debug!("vera: describe: photo {} not embedded: {e}", job.photo.0);
                            None
                        }
                        (Ok(_), None) => None,
                    };
                    made.clone_from(&v);
                    v
                }
            };
            vector.and_then(|v| super::keywords::score(&k.vocab, &k.matrix, k.calibration.as_deref(), &v, k.limit))
        }
        None => None,
    };
    let Some(remote) = job.remote.take() else {
        let outcome = match local {
            Some(l) => Ok(Outcome { described: None, local: Some(l), model: None, host: None }),
            None => Err("no keywords could be proposed: the photo has no embedding".into()),
        };
        return (outcome, made);
    };
    let Some(image) = image else { return (Err("the photo could not be rendered".into()), made) };
    let jpeg = match lightcraft_codecs::encode_jpeg(
        &lightcraft_codecs::EncodeImage::rgba8(&image),
        JPEG_QUALITY,
        lightcraft_codecs::ChromaSubsampling::S420,
        &lightcraft_codecs::EncodeMeta::default(),
    ) {
        Ok(j) => j,
        Err(e) => return (Err(format!("the photo could not be encoded: {e}")), made),
    };
    drop(image);
    let b64 = base64(&jpeg);
    drop(jpeg);
    let outcome = describe(&remote, &b64, shared, &job.cancel).map(|d| Outcome {
        described: Some(d),
        local,
        model: Some(remote.model.clone()),
        host: Some(remote.host.clone()),
    });
    (outcome, made)
}

/// The error of a job stopped by Cancel (not listed as a failure).
pub(crate) const CANCELLED: &str = "cancelled";

/// Ask the endpoint for one photo's description, with the retry rules of the module doc.
fn describe(remote: &Remote, b64: &str, shared: &Shared, cancel: &AtomicBool) -> Result<Described, String> {
    let mut attempts = 0u32;
    let mut limited = 0u32;
    let mut last;
    loop {
        if !shared.wait_ready(cancel) {
            return Err(CANCELLED.into());
        }
        let format = Format::from_u8(shared.format.load(Ordering::Relaxed));
        let body = request_body(&remote.model, b64, format).to_string();
        let mut headers = vec![("Content-Type", "application/json".to_string())];
        if let Some(t) = &remote.token {
            headers.push(("Authorization", format!("Bearer {t}")));
        }
        let limits = Limits { connect: CONNECT, stall: STALL, cancel };
        let answer = http::request("POST", &remote.url, &headers, Some(body.as_bytes()), &limits).and_then(|mut r| {
            let wait = retry_after(r.header("retry-after"));
            let body = r.read_to_end(MAX_RESPONSE, &limits)?;
            Ok((r.status, wait, body))
        });
        match answer {
            Ok((200..=299, _, body)) => match parse_response(&body) {
                Ok(d) => return Ok(d),
                Err(e) => {
                    attempts += 1;
                    last = e;
                }
            },
            Ok((400, _, _)) if format != Format::None => {
                // the endpoint does not take this response format: one step down, for every request after this one
                let next = if format == Format::Schema { Format::JsonObject } else { Format::None };
                let _ = shared.format.compare_exchange(format as u8, next as u8, Ordering::Relaxed, Ordering::Relaxed);
                log::info!("vera: describe: {} refused response_format {:?}; trying {:?}", remote.host, format, next);
                continue;
            }
            Ok((429, wait, _)) => {
                limited += 1;
                if limited > MAX_RATE_LIMITED {
                    return Err(format!("the endpoint kept saying it is busy (HTTP 429, {limited} times)"));
                }
                log::debug!("vera: describe: {} is busy; waiting {} s", remote.host, wait.as_secs());
                shared.pause_for(wait);
                continue;
            }
            Ok((status @ (401 | 403), _, _)) => {
                return Err(format!("the endpoint refused the request (HTTP {status}): check the token with vera.describe.config"));
            }
            Ok((status, _, body)) => {
                attempts += 1;
                let snippet: String = String::from_utf8_lossy(&body).chars().take(200).collect();
                last = remote.scrub(&format!("HTTP {status}: {}", snippet.trim()));
            }
            Err(HttpError::Cancelled) => return Err(CANCELLED.into()),
            Err(e) => {
                attempts += 1;
                last = remote.scrub(&e.to_string());
            }
        }
        if attempts >= ATTEMPTS {
            return Err(format!("{last} (after {attempts} attempts)"));
        }
        // a short pause before the next attempt, still cancellable
        let until = Instant::now() + BACKOFF * attempts;
        while Instant::now() < until {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_follow_the_policy() {
        assert!(endpoint("http://127.0.0.1:8080", false).is_ok(), "http to loopback");
        assert!(endpoint("http://localhost:11434/v1", false).is_ok());
        assert!(endpoint("http://[::1]:1234/", false).is_ok());
        assert!(endpoint("https://localhost:8443", false).is_ok());
        let e = endpoint("http://192.168.1.20:8080", true).unwrap_err();
        assert!(e.contains("https://"), "http to another computer is refused even with consent: {e}");
        let e = endpoint("https://api.example.com/v1", false).unwrap_err();
        assert!(e.contains("allowRemoteHost"), "{e}");
        assert!(endpoint("https://api.example.com/v1", true).is_ok(), "https with consent");
        for bad in ["", "ftp://x", "localhost:8080", "https://", "http://u:p@127.0.0.1/", "http://127.0.0.1/v1?key=abc"] {
            assert!(endpoint(bad, true).is_err(), "{bad}");
        }
        assert!(endpoint(&format!("https://h.example/{}", "a".repeat(MAX_ENDPOINT)), true).is_err());
    }

    #[test]
    fn completions_urls() {
        let at = |s: &str| completions_url(&Url::parse(s).unwrap()).path;
        assert_eq!(at("http://127.0.0.1:8080"), "/v1/chat/completions");
        assert_eq!(at("http://127.0.0.1:8080/"), "/v1/chat/completions");
        assert_eq!(at("http://127.0.0.1:8080/v1"), "/v1/chat/completions");
        assert_eq!(at("http://127.0.0.1:8080/v1/"), "/v1/chat/completions");
        assert_eq!(at("https://h.example/openai/v1/chat/completions"), "/openai/v1/chat/completions");
        assert_eq!(at("https://h.example/proxy"), "/proxy/v1/chat/completions");
        assert_eq!(host_of(&Url::parse("http://127.0.0.1:8080/x").unwrap()), "127.0.0.1:8080");
        assert_eq!(host_of(&Url::parse("https://h.example/x").unwrap()), "h.example");
        assert_eq!(host_of(&Url::parse("http://[::1]:9/").unwrap()), "[::1]:9");
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lc-vera-remote-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_token_file_is_private_and_the_environment_overrides_it() {
        let dir = temp_dir("token");
        let models = dir.join("models");
        let path = token_path(Some(&models.join("siglip2"))).unwrap();
        assert_eq!(path, models.join(TOKEN_FILE));
        assert_eq!(read_token(Some(&path), None).unwrap(), None, "no file, no token");
        write_token(&path, Some("sk-test-123")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert_eq!(read_token(Some(&path), None).unwrap().as_deref(), Some("sk-test-123"));
        // replaced atomically, still private
        write_token(&path, Some("sk-second")).unwrap();
        assert_eq!(read_token(Some(&path), None).unwrap().as_deref(), Some("sk-second"));
        assert!(!path.with_extension("token.part").exists());
        // the environment wins
        assert_eq!(read_token(Some(&path), Some("from-env".into())).unwrap().as_deref(), Some("from-env"));
        assert_eq!(read_token(Some(&path), Some("  ".into())).unwrap().as_deref(), Some("sk-second"), "a blank variable is unset");
        assert!(read_token(Some(&path), Some("two words".into())).is_err());
        assert_eq!(read_token(None, Some("env-only".into())).unwrap().as_deref(), Some("env-only"));
        // a file others can read is refused, and says so without its content
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
            let e = read_token(Some(&path), None).unwrap_err();
            assert!(e.contains("other users") && !e.contains("sk-second"), "{e}");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        // hostile tokens are refused before they reach a header
        for bad in ["", "a\r\nX-Injected: 1", "with space", &"x".repeat(MAX_TOKEN + 1)] {
            assert!(write_token(&path, Some(bad)).is_err(), "{bad:?}");
        }
        write_token(&path, None).unwrap();
        assert!(!path.exists());
        write_token(&path, None).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn responses_are_parsed_strictly_and_leniently() {
        let wrap = |content: &str| json!({"choices": [{"message": {"role": "assistant", "content": content}}]}).to_string();
        let d = parse_response(
            wrap(r#"{"title":"Lake at Dawn.","caption":"A still lake.","alt":"A lake under peaks.","keywords":["Lake","mountains","lake","a|b"]}"#)
                .as_bytes(),
        )
        .unwrap();
        assert_eq!(d.title, "Lake at Dawn");
        assert_eq!(d.caption, "A still lake.");
        assert_eq!(d.alt_text, "A lake under peaks.");
        assert_eq!(d.keywords, ["lake", "mountains", "a b"]);
        // a fenced answer, and one with prose around it
        let fenced = wrap("```json\n{\"title\":\"T\",\"caption\":\"C\",\"alt\":\"A\",\"keywords\":[]}\n```");
        assert_eq!(parse_response(fenced.as_bytes()).unwrap().title, "T");
        let prose = wrap("Sure! Here it is: {\"title\": \"Brace } in text\", \"keywords\": \"sea, sky\"} Hope it helps.");
        let d = parse_response(prose.as_bytes()).unwrap();
        assert_eq!((d.title.as_str(), d.keywords.len()), ("Brace } in text", 2));
        // lengths are capped at character boundaries
        let long = wrap(&json!({"title": "é".repeat(500), "keywords": (0..50).map(|i| format!("k{i}")).collect::<Vec<_>>()}).to_string());
        let d = parse_response(long.as_bytes()).unwrap();
        assert_eq!((d.title.chars().count(), d.keywords.len()), (MAX_TITLE, MAX_REMOTE_KEYWORDS));
        // wrong types and empty answers are errors
        for bad in [
            wrap(r#"{"title": 5}"#),
            wrap(r#"{"keywords": [1, 2]}"#),
            wrap(r#"{"keywords": {"a": 1}}"#),
            wrap("no json here"),
            wrap("{}"),
            wrap("[1, 2]"),
            "not json".to_string(),
            json!({"choices": []}).to_string(),
            json!({"choices": [{"message": {"content": 7}}]}).to_string(),
        ] {
            assert!(parse_response(bad.as_bytes()).is_err(), "{bad}");
        }
        // a list of content parts
        let parts = json!({"choices": [{"message": {"content": [{"type": "text", "text": "{\"caption\": \"Parts.\"}"}]}}]}).to_string();
        assert_eq!(parse_response(parts.as_bytes()).unwrap().caption, "Parts.");
    }

    #[test]
    fn the_request_has_the_image_and_the_format() {
        let b = request_body("llava", "QUJD", Format::Schema);
        assert_eq!(b["model"], "llava");
        assert_eq!(b["temperature"], 0.2);
        assert_eq!(b["max_tokens"], 400);
        assert_eq!(b["response_format"]["type"], "json_schema");
        assert_eq!(b["response_format"]["json_schema"]["strict"], true);
        assert_eq!(b["messages"][1]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,QUJD");
        assert_eq!(request_body("m", "x", Format::JsonObject)["response_format"], json!({"type": "json_object"}));
        assert!(request_body("m", "x", Format::None).get("response_format").is_none());
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
        assert_eq!(base64(b""), "");
        assert_eq!(retry_after(Some("3")), Duration::from_secs(3));
        assert_eq!(retry_after(Some("99999")), MAX_RETRY_AFTER);
        assert_eq!(retry_after(Some("Wed, 21 Oct 2026 07:28:00 GMT")), DEFAULT_RETRY_AFTER);
        assert_eq!(retry_after(None), DEFAULT_RETRY_AFTER);
    }
}
