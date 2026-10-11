//! The Describe lane on a session: the endpoint policy and the token, the worker against a stub OpenAI-compatible
//! server on loopback (no network), the review queue and `describe.accept`, zero-shot keywords with hand-made vectors,
//! and (gated on `LIGHTCRAFT_SIGLIP2_DIR`) the real vocabulary on the demo library.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lightcraft_catalog::{PhotoId, Provenance};
use lightcraft_embed::DIM;
use lightcraft_vera::cutoff::mix64;
use serde_json::{Value, json};

use super::keywords::Matrix;
use crate::Session;
use crate::media::content_key;

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lc-vera-describe-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn random_vec(seed: u64) -> Vec<f32> {
    let v: Vec<f32> = (0..DIM as u64).map(|i| (mix64(seed.wrapping_mul(0x9E37_79B9) ^ i) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32).collect();
    unit(&v)
}

/// Photos of the demo library that show distinct pictures, in id order.
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

// ---- a stub OpenAI-compatible server on loopback

/// One request the stub saw.
#[derive(Clone)]
struct Seen {
    at: Instant,
    head: String,
    body: String,
}

/// What the stub answers to its `i`-th request: (status, extra headers, body, delay before answering).
type Answer = Box<dyn Fn(usize, &str) -> (u16, Vec<(String, String)>, String, Duration) + Send + Sync>;

struct Stub {
    base: String,
    /// Each request, with when its answer was sent.
    seen: Arc<Mutex<Vec<(Seen, Instant)>>>,
    /// Requests that have arrived (answered or not).
    arrived: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}

impl Stub {
    fn start(answer: Answer) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen: Arc<Mutex<Vec<(Seen, Instant)>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let answer = Arc::new(answer);
        let count = Arc::new(AtomicUsize::new(0));
        let (seen2, stop2, count2) = (seen.clone(), stop.clone(), count.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                if stop2.load(Ordering::Relaxed) {
                    return;
                }
                let Ok(mut sock) = conn else { continue };
                let (answer, seen, count) = (answer.clone(), seen2.clone(), count2.clone());
                // each connection on its own thread, so requests in flight overlap as they would on a real server
                std::thread::spawn(move || {
                    let mut data = Vec::new();
                    let mut buf = [0u8; 65536];
                    let head_end = loop {
                        let Ok(n) = sock.read(&mut buf) else { return };
                        if n == 0 {
                            return;
                        }
                        data.extend_from_slice(&buf[..n]);
                        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                            break i + 4;
                        }
                    };
                    let at = Instant::now();
                    let head = String::from_utf8_lossy(&data[..head_end]).to_string();
                    let len = head
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    while data.len() < head_end + len {
                        let Ok(n) = sock.read(&mut buf) else { return };
                        if n == 0 {
                            return;
                        }
                        data.extend_from_slice(&buf[..n]);
                    }
                    let body = String::from_utf8_lossy(&data[head_end..head_end + len]).to_string();
                    let i = count.fetch_add(1, Ordering::SeqCst);
                    let (status, headers, reply, delay) = answer(i, &body);
                    std::thread::sleep(delay);
                    let mut out = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n", reply.len());
                    for (k, v) in headers {
                        out.push_str(&format!("{k}: {v}\r\n"));
                    }
                    out.push_str("\r\n");
                    out.push_str(&reply);
                    seen.lock().unwrap().push((Seen { at, head, body }, Instant::now()));
                    let _ = sock.write_all(out.as_bytes());
                });
            }
        });
        Stub { base, seen, arrived: count, stop }
    }

    fn host(&self) -> &str {
        self.base.trim_start_matches("http://")
    }

    fn seen(&self) -> Vec<Seen> {
        let mut v: Vec<(Seen, Instant)> = self.seen.lock().unwrap().clone();
        v.sort_by_key(|(s, _)| s.at);
        v.into_iter().map(|(s, _)| s).collect()
    }

    /// (arrival, answer) of every request.
    fn answered(&self) -> Vec<(Instant, Instant)> {
        self.seen.lock().unwrap().iter().map(|(s, t)| (s.at, *t)).collect()
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(self.host());
    }
}

/// A chat-completions answer holding `content`.
fn completion(content: &Value) -> String {
    json!({"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": content.to_string()}}]})
        .to_string()
}

fn good() -> String {
    completion(&json!({
        "title": "Lake Below the Peaks",
        "caption": "A still alpine lake at first light.",
        "alt": "A calm lake reflecting snowy mountains under a pale sky.",
        "keywords": ["lake", "mountains", "Dawn"],
    }))
}

fn ok_answer(_: usize, _: &str) -> (u16, Vec<(String, String)>, String, Duration) {
    (200, Vec::new(), good(), Duration::ZERO)
}

/// A demo session pointed at the stub (loopback: no token needed).
fn session_with(stub: &Stub) -> Session {
    let mut s = Session::with_demo();
    let r = s.execute("vera.describe.config", &json!({"endpoint": stub.base, "model": "stub-vlm"})).unwrap();
    assert_eq!(r["state"], "live", "{r}");
    s
}

/// Pump until the Describe lane has nothing pending or in flight.
fn drain(s: &mut Session, secs: u64) -> Value {
    let started = Instant::now();
    loop {
        let p = s.execute("vera.pump", &json!({"pace": "normal"})).unwrap();
        let d = &p["describe"];
        if d["pending"] == 0 && d["inFlight"] == 0 {
            return p;
        }
        assert!(started.elapsed().as_secs() < secs, "describing did not finish: {p}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn proposals(s: &mut Session) -> Value {
    s.execute("describe.list", &json!({})).unwrap()
}

// ---- configuration, policy and the token

#[test]
fn describe_config_follows_the_endpoint_policy() {
    let mut s = Session::with_demo();
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(st["lanes"]["describe"]["state"], "needs_endpoint", "{st}");
    assert!(st["lanes"]["describe"]["reason"].as_str().unwrap().contains("vera.describe.config"));
    // http to this computer is fine; the model name is still needed
    let r = s.execute("vera.describe.config", &json!({"endpoint": "http://127.0.0.1:8080"})).unwrap();
    assert_eq!((r["state"].as_str(), r["tokenSet"].as_bool()), (Some("needs_endpoint"), Some(false)), "{r}");
    let r = s.execute("vera.describe.config", &json!({"model": "llava"})).unwrap();
    assert_eq!(r["state"], "live", "{r}");
    assert_eq!(r["endpoint"], "http://127.0.0.1:8080");
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["lanes"]["describe"]["state"].as_str(), st["lanes"]["describe"]["host"].as_str()), (Some("live"), Some("127.0.0.1:8080")));
    // plain http to another computer: refused, consent or not, and nothing changes
    for p in [json!({"endpoint": "http://192.168.1.5:8080"}), json!({"endpoint": "http://example.com", "allowRemoteHost": true})] {
        let e = s.execute("vera.describe.config", &p).unwrap_err().to_string();
        assert!(e.contains("https://"), "{e}");
    }
    // https to another computer: refused without consent, accepted with it
    let e = s.execute("vera.describe.config", &json!({"endpoint": "https://vlm.example.com/v1"})).unwrap_err().to_string();
    assert!(e.contains("allowRemoteHost"), "{e}");
    assert_eq!(s.execute("vera.describe.config", &json!({})).unwrap()["endpoint"], "http://127.0.0.1:8080", "unchanged");
    let r = s.execute("vera.describe.config", &json!({"endpoint": "https://vlm.example.com/v1", "allowRemoteHost": true})).unwrap();
    assert_eq!(r["state"], "needs_token", "a remote host needs a token: {r}");
    // taking the consent back with a remote endpoint in place is refused too
    let e = s.execute("vera.describe.config", &json!({"allowRemoteHost": false})).unwrap_err().to_string();
    assert!(e.contains("allowRemoteHost"), "{e}");
    // hostile params
    for p in [json!({"endpoint": 5}), json!({"allowRemoteHost": "yes"}), json!({"model": "a\nb"}), json!({"token": 7}), json!({"token": "a b"})] {
        assert!(s.execute("vera.describe.config", &p).is_err(), "{p}");
    }
    // this session has nowhere to keep a token
    let e = s.execute("vera.describe.config", &json!({"token": "sk-1"})).unwrap_err().to_string();
    assert!(e.contains("nowhere"), "{e}");
    // clearing the endpoint
    let r = s.execute("vera.describe.config", &json!({"endpoint": null})).unwrap();
    assert_eq!((r["endpoint"].is_null(), r["state"].as_str()), (true, Some("needs_endpoint")));
    // photo.describe says what is missing
    let id = distinct(&s)[0];
    let e = s.execute("photo.describe", &json!({"ids": [id.0]})).unwrap_err().to_string();
    assert!(e.contains("endpoint"), "{e}");
}

#[test]
fn describe_token_is_private_and_never_shown() {
    const SECRET: &str = "sk-very-secret-4711";
    let dir = temp_dir("token");
    let lib = dir.join("lib");
    let models = dir.join("models");
    let mut s = Session::new();
    s.open_library(&lib, true).unwrap();
    s.vera_model_dir = Some(models.join(super::lane::MODEL_ID));
    let r = s
        .execute("vera.describe.config", &json!({"endpoint": "https://vlm.example.com", "model": "m", "allowRemoteHost": true, "token": SECRET}))
        .unwrap();
    assert_eq!((r["tokenSet"].as_bool(), r["state"].as_str()), (Some(true), Some("live")), "{r}");
    assert!(!r.to_string().contains(SECRET), "never in the output: {r}");
    let path = models.join(super::remote::TOKEN_FILE);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), SECRET);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // not in the status, the journal, the preferences or the catalog
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert!(!st.to_string().contains(SECRET));
    assert!(!s.journal.iter().any(|(id, p)| id == "vera.describe.config" || p.to_string().contains(SECRET)), "not journaled");
    let prefs = std::fs::read_to_string(lib.join("prefs.json")).unwrap();
    assert!(!prefs.contains(SECRET), "{prefs}");
    let saved: Value = serde_json::from_str(&prefs).unwrap();
    assert_eq!(saved["vera"]["describe"], json!({"endpoint": "https://vlm.example.com", "model": "m", "allowRemoteHost": true}), "{prefs}");
    // a token file others can read is refused
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let r = s.execute("vera.describe.config", &json!({})).unwrap();
        assert_eq!((r["state"].as_str(), r["tokenSet"].as_bool()), (Some("error"), Some(false)), "{r}");
        assert!(r["reason"].as_str().unwrap().contains("other users") && !r.to_string().contains(SECRET), "{r}");
        let e = s.execute("photo.describe", &json!({"ids": [distinct(&s)[0].0]})).unwrap_err().to_string();
        assert!(e.contains("other users") && !e.contains(SECRET), "{e}");
        // setting it again writes it privately
        s.execute("vera.describe.config", &json!({"token": SECRET})).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // the preferences come back with the library; the token is removed with null
    s.close_library().unwrap();
    drop(s);
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    s.vera_model_dir = Some(models.join(super::lane::MODEL_ID));
    let r = s.execute("vera.describe.config", &json!({})).unwrap();
    assert_eq!((r["endpoint"].as_str(), r["tokenSet"].as_bool()), (Some("https://vlm.example.com"), Some(true)), "{r}");
    let r = s.execute("vera.describe.config", &json!({"token": null})).unwrap();
    assert_eq!((r["tokenSet"].as_bool(), r["state"].as_str()), (Some(false), Some("needs_token")), "{r}");
    assert!(!path.exists());
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- the worker against the stub

#[test]
fn describe_a_proposal_lands_in_the_review_queue() {
    let stub = Stub::start(Box::new(ok_answer));
    let mut s = session_with(&stub);
    let ids = distinct(&s);
    let (a, b) = (ids[0], ids[1]);
    let before = s.catalog.photo(a).unwrap().meta.clone();
    let r = s.execute("photo.describe", &json!({"ids": [a.0, b.0, a.0, 999_999]})).unwrap();
    assert_eq!((r["queued"].as_u64(), r["host"].as_str()), (Some(2), Some(stub.host())), "{r}");
    assert_eq!(r["skipped"].as_array().unwrap().len(), 1, "{r}");
    let row = s.activity.list().into_iter().find(|t| t.kind == "describe").expect("an activity row");
    assert_eq!(row.label, "Describing photos");
    assert!(row.cancellable && row.detail.contains("127.0.0.1"), "{row:?}");
    drain(&mut s, 60);
    assert!(!s.activity.list().iter().any(|t| t.kind == "describe"), "the row goes when the work is done");
    let l = proposals(&mut s);
    assert_eq!(l["total"], 2, "{l}");
    let p = l["proposals"].as_array().unwrap().iter().find(|p| p["photo"] == a.0).unwrap().clone();
    assert_eq!(p["fields"]["title"], "Lake Below the Peaks");
    assert_eq!(p["fields"]["caption"], "A still alpine lake at first light.");
    assert_eq!(p["fields"]["alt_text"], "A calm lake reflecting snowy mountains under a pale sky.");
    assert_eq!(p["fields"]["keywords"], json!(["lake", "mountains", "dawn"]));
    assert_eq!(p["source"], json!({"kind": "remote", "model": "stub-vlm", "host": stub.host()}));
    assert_eq!(p["current"]["title"], before.title, "the photo's own values beside the proposal: {p}");
    assert_eq!(s.catalog.photo(a).unwrap().meta, before, "nothing is written to the photo");
    // what was sent: the model, the strict schema, a JPEG data URL, no token
    let seen = stub.seen();
    assert_eq!(seen.len(), 2);
    assert!(seen[0].head.starts_with("POST /v1/chat/completions HTTP/1.1"), "{}", seen[0].head);
    assert!(!seen[0].head.to_ascii_lowercase().contains("authorization"), "no token to loopback when none is set");
    let body: Value = serde_json::from_str(&seen[0].body).unwrap();
    assert_eq!((body["model"].as_str(), body["response_format"]["type"].as_str()), (Some("stub-vlm"), Some("json_schema")));
    let url = body["messages"][1]["content"][1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/jpeg;base64,/9j/"), "a JPEG");
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!((st["lanes"]["describe"]["proposals"].as_u64(), st["lanes"]["describe"]["pending"].as_u64()), (Some(2), Some(0)), "{st}");
    // rejected: gone, nothing written
    let r = s.execute("describe.reject", &json!({"id": b.0})).unwrap();
    assert_eq!((r["removed"].as_bool(), r["proposals"].as_u64()), (Some(true), Some(1)));
    assert!(!s.execute("describe.reject", &json!({"id": b.0})).unwrap()["removed"].as_bool().unwrap());
    let r = s.execute("describe.clear", &json!({})).unwrap();
    assert_eq!(r["removed"], 1);
    assert_eq!(proposals(&mut s)["total"], 0);
}

#[test]
fn describe_a_400_steps_down_the_response_format() {
    let stub = Stub::start(Box::new(|_, body: &str| {
        if body.contains("json_schema") {
            (400, Vec::new(), json!({"error": "response_format json_schema is not supported"}).to_string(), Duration::ZERO)
        } else {
            (200, Vec::new(), good(), Duration::ZERO)
        }
    }));
    let mut s = session_with(&stub);
    let a = distinct(&s)[0];
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    assert_eq!(proposals(&mut s)["total"], 1);
    let formats: Vec<Value> =
        stub.seen().iter().map(|r| serde_json::from_str::<Value>(&r.body).unwrap()["response_format"]["type"].clone()).collect();
    assert_eq!(formats, [json!("json_schema"), json!("json_object")]);
    // the step down is remembered: the next photo asks in JSON mode at once
    let b = distinct(&s)[1];
    s.execute("photo.describe", &json!({"ids": [b.0]})).unwrap();
    drain(&mut s, 60);
    assert_eq!(stub.seen().len(), 3);
    // an endpoint that takes no format at all: the first {…} of its prose is read
    let prose = Stub::start(Box::new(|_, body: &str| {
        if body.contains("response_format") {
            (400, Vec::new(), "{}".into(), Duration::ZERO)
        } else {
            let content =
                "Here you go:\n```json\n{\"title\": \"Dunes\", \"caption\": \"Sand.\", \"alt\": \"Sand dunes.\", \"keywords\": [\"desert\"]}\n```";
            (200, Vec::new(), json!({"choices": [{"message": {"content": content}}]}).to_string(), Duration::ZERO)
        }
    }));
    let mut s = session_with(&prose);
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    let l = proposals(&mut s);
    assert_eq!(l["proposals"][0]["fields"]["title"], "Dunes", "{l}");
    assert_eq!(prose.seen().len(), 3);
}

#[test]
fn describe_a_429_pauses_every_request_and_spends_no_attempt() {
    // three 429s for the first requests, then answers: a photo with three attempts still gets its proposal
    let stub = Stub::start(Box::new(|i, _| {
        if i < 3 { (429, vec![("Retry-After".into(), "1".into())], "{}".into(), Duration::ZERO) } else { (200, Vec::new(), good(), Duration::ZERO) }
    }));
    let mut s = session_with(&stub);
    let ids = distinct(&s);
    s.execute("photo.describe", &json!({"ids": [ids[0].0]})).unwrap();
    drain(&mut s, 60);
    let l = proposals(&mut s);
    assert_eq!((l["total"].as_u64(), l["failed"].as_array().unwrap().len()), (Some(1), 0), "{l}");
    assert_eq!(stub.seen().len(), 4);
    // one 429 holds back the other threads' next requests too
    let stub = Stub::start(Box::new(|i, _| {
        if i == 0 { (429, vec![("Retry-After".into(), "1".into())], "{}".into(), Duration::ZERO) } else { (200, Vec::new(), good(), Duration::ZERO) }
    }));
    let mut s = session_with(&stub);
    s.execute("photo.describe", &json!({"ids": ids.iter().take(6).map(|i| i.0).collect::<Vec<_>>()})).unwrap();
    drain(&mut s, 60);
    assert_eq!(proposals(&mut s)["total"], 6);
    let answered = stub.answered();
    let limited_at = answered.iter().min_by_key(|(at, _)| *at).unwrap().1;
    // requests that arrived once the 429 had been answered (and so were sent after it) waited for its Retry-After
    let late: Vec<Duration> = answered.iter().filter(|(at, _)| *at > limited_at).map(|(at, _)| at.duration_since(limited_at)).collect();
    assert!(!late.is_empty());
    assert!(late.iter().all(|d| *d >= Duration::from_millis(900)), "{late:?}");
}

#[test]
fn describe_three_failures_fail_the_photo() {
    let stub = Stub::start(Box::new(|_, _| (500, Vec::new(), json!({"error": "model crashed"}).to_string(), Duration::ZERO)));
    let mut s = session_with(&stub);
    let a = distinct(&s)[0];
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    let l = proposals(&mut s);
    assert_eq!(l["total"], 0);
    let failed = l["failed"].as_array().unwrap();
    assert_eq!(failed.len(), 1, "{l}");
    assert_eq!(failed[0]["photo"], a.0);
    let e = failed[0]["error"].as_str().unwrap();
    assert!(e.contains("HTTP 500") && e.contains("3 attempts"), "{e}");
    assert_eq!(stub.seen().len(), 3);
    // a refused token fails at once and stops the lane until the settings change
    let stub = Stub::start(Box::new(|_, _| (401, Vec::new(), "{}".into(), Duration::ZERO)));
    let mut s = session_with(&stub);
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    assert_eq!(stub.seen().len(), 1, "no retries for a refused token");
    let st = s.execute("vera.status", &json!({})).unwrap();
    assert_eq!(st["lanes"]["describe"]["state"], "error", "{st}");
    assert!(s.execute("photo.describe", &json!({"ids": [a.0]})).is_err());
    let r = s.execute("vera.describe.config", &json!({"model": "other"})).unwrap();
    assert_eq!(r["state"], "live", "a new setting clears it: {r}");
}

#[test]
fn describe_cancel_stops_new_requests() {
    let stub = Stub::start(Box::new(|_, _| (200, Vec::new(), good(), Duration::from_millis(500))));
    let mut s = session_with(&stub);
    let ids: Vec<u64> = distinct(&s).iter().take(12).map(|i| i.0).collect();
    assert_eq!(ids.len(), 12);
    s.execute("photo.describe", &json!({"ids": ids})).unwrap();
    // the first requests are on their way
    let started = Instant::now();
    while stub.arrived.load(Ordering::SeqCst) == 0 && started.elapsed().as_secs() < 30 {
        s.execute("vera.pump", &json!({"pace": "normal"})).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
    let row = s.activity.list().into_iter().find(|t| t.kind == "describe").unwrap();
    s.execute("activity.cancel", &json!({"id": row.id})).unwrap();
    drain(&mut s, 30);
    std::thread::sleep(Duration::from_millis(800));
    s.execute("vera.pump", &json!({"pace": "normal"})).unwrap();
    let sent = stub.seen().len();
    assert!(sent <= super::remote::THREADS, "{sent} requests: none after the cancel");
    let l = proposals(&mut s);
    assert_eq!(
        (l["total"].as_u64(), l["failed"].as_array().unwrap().len()),
        (Some(0), 0),
        "cancelled photos are neither proposals nor failures: {l}"
    );
    assert!(super::describe::queued(&s).is_empty());
    assert!(!s.activity.list().iter().any(|t| t.kind == "describe"));
    // pause holds back new requests; a new run starts afresh
    s.execute("photo.describe", &json!({"ids": [ids[0]]})).unwrap();
    for _ in 0..20 {
        s.execute("vera.pump", &json!({"pace": "pause"})).unwrap();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(stub.seen().len(), sent, "nothing sent while paused");
    drain(&mut s, 30);
    assert_eq!(proposals(&mut s)["total"], 1);
}

// ---- accepting

#[test]
fn describe_accept_writes_meta_with_provenance_in_one_undo_step() {
    let stub = Stub::start(Box::new(ok_answer));
    let mut s = session_with(&stub);
    let a = distinct(&s)[0];
    let before = s.catalog.photo(a).unwrap().meta.clone();
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    // hostile accepts
    for p in [
        json!({}),
        json!({"id": "x"}),
        json!({"id": a.0, "fields": ["colour"]}),
        json!({"id": a.0, "edits": {"title": 3}}),
        json!({"id": a.0, "edits": {"gps": "1"}}),
    ] {
        assert!(s.execute("describe.accept", &p).is_err(), "{p}");
    }
    assert!(s.execute("describe.accept", &json!({"id": distinct(&s)[1].0})).is_err(), "no proposal");
    let undo_before = s.undo.len();
    let r = s.execute("describe.accept", &json!({"id": a.0, "edits": {"caption": "My own caption."}})).unwrap();
    assert_eq!(r["accepted"], json!(["title", "caption", "alt_text", "keywords"]), "{r}");
    assert_eq!(s.undo.len(), undo_before + 1, "one undo step");
    assert_eq!(s.undo.last().unwrap().label, "Accept Description");
    let m = s.catalog.photo(a).unwrap().meta.clone();
    assert_eq!(m.title, "Lake Below the Peaks");
    assert_eq!(m.caption, "My own caption.");
    assert_eq!(m.alt_text, "A calm lake reflecting snowy mountains under a pale sky.");
    // keywords are merged with the photo's own
    for k in before.keywords.iter().map(String::as_str).chain(["lake", "mountains", "dawn"]) {
        assert!(m.keywords.iter().any(|x| x.eq_ignore_ascii_case(k)), "{k}: {:?}", m.keywords);
    }
    for f in ["title", "caption", "alt_text", "keywords"] {
        match m.provenance.get(f) {
            Some(Provenance::Describe { model, at }) => assert!(model == "stub-vlm" && !at.is_empty()),
            other => panic!("{f}: {other:?}"),
        }
    }
    assert_eq!(proposals(&mut s)["total"], 0, "the proposal leaves the queue");
    // a later plain edit leaves the provenance alone; a provenance param sets it
    s.execute("photo.setMeta", &json!({"ids": [a.0], "title": "Typed"})).unwrap();
    assert!(matches!(s.catalog.photo(a).unwrap().meta.provenance.get("title"), Some(Provenance::Describe { .. })));
    s.execute("photo.setMeta", &json!({"ids": [a.0], "provenance": {"title": {"kind": "user"}, "caption": null}})).unwrap();
    let m2 = s.catalog.photo(a).unwrap().meta.clone();
    assert_eq!((m2.provenance.get("title"), m2.provenance.get("caption")), (Some(&Provenance::User), None));
    assert!(s.execute("photo.setMeta", &json!({"ids": [a.0], "provenance": {"gps": {"kind": "user"}}})).is_err());
    assert!(s.execute("photo.setMeta", &json!({"ids": [a.0], "provenance": {"title": {"kind": "oracle"}}})).is_err());
    // undo goes back to before the accept
    for _ in 0..3 {
        s.execute("edit.undo", &json!({})).unwrap();
    }
    assert_eq!(s.catalog.photo(a).unwrap().meta, before, "undone");
    // replaceKeywords replaces, and a subset of fields is a subset
    let b = distinct(&s)[1];
    let caption_b = s.catalog.photo(b).unwrap().meta.caption.clone();
    s.execute("photo.describe", &json!({"ids": [b.0]})).unwrap();
    drain(&mut s, 60);
    s.execute("describe.accept", &json!({"id": b.0, "fields": ["title", "keywords"], "replaceKeywords": true})).unwrap();
    let m = s.catalog.photo(b).unwrap().meta.clone();
    assert_eq!(m.keywords, ["lake", "mountains", "dawn"]);
    assert_eq!(m.caption, caption_b);
    assert!(m.provenance.contains_key("title") && !m.provenance.contains_key("caption"));
}

#[test]
fn describe_accept_writes_the_contextual_and_fused_roles() {
    let stub = Stub::start(Box::new(ok_answer));
    let mut s = session_with(&stub);
    let ids = distinct(&s);
    for (i, id) in ids.iter().enumerate() {
        let key = content_key(s.catalog.photo(*id).unwrap());
        assert!(s.vera.lane.store.insert(key, &random_vec(500 + i as u64)));
    }
    let a = ids[0];
    s.execute("photo.describe", &json!({"ids": [a.0]})).unwrap();
    drain(&mut s, 60);
    s.execute("describe.accept", &json!({"id": a.0})).unwrap();
    let m = s.catalog.photo(a).unwrap().meta.clone();
    let text = super::review::contextual_text(&m);
    assert!(text.starts_with("Lake Below the Peaks. A still alpine lake at first light. "), "{text}");
    let ctx = random_vec(9999);
    s.vera.lane.test_queries.insert(text, ctx.clone());
    s.execute("vera.pump", &json!({"pace": "normal"})).unwrap();
    let key = content_key(s.catalog.photo(a).unwrap());
    let store = &s.vera.lane.store;
    let got = store.get_role(&key, super::embeddings::ROLE_CONTEXTUAL).expect("contextual written");
    assert!(lightcraft_vera::cutoff::cosine(got, &ctx) > 0.9999);
    let fused = store.get_role(&key, super::embeddings::ROLE_FUSED).expect("fused written");
    let canonical = store.get(&key).unwrap();
    let want = unit(&canonical.iter().zip(&ctx).map(|(a, b)| a + b).collect::<Vec<_>>());
    assert!(lightcraft_vera::cutoff::cosine(fused, &want) > 0.9999);
    assert_eq!((store.role_count(super::embeddings::ROLE_CONTEXTUAL), store.role_count(super::embeddings::ROLE_FUSED)), (1, 1));
    assert!(s.vera.describe.contextual.is_empty());
    // photo.similar uses the fused vector once there is one (a folder that looks like a model makes the lane live; the
    // photo's vector is already stored, so nothing is loaded)
    let dir = temp_dir("fused");
    for f in [lightcraft_embed::WEIGHTS_FILE, lightcraft_embed::CONFIG_FILE, lightcraft_embed::TOKENIZER_FILE] {
        std::fs::write(dir.join(f), b"not a model").unwrap();
    }
    s.vera_model_dir = Some(dir.clone());
    let r = s.execute("photo.similar", &json!({"id": a.0, "filter": false})).unwrap();
    assert_eq!(r["role"], "fused", "{r}");
    let r = s.execute("photo.similar", &json!({"id": ids[1].0, "filter": false})).unwrap();
    assert_eq!(r["role"], "canonical", "{r}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_proposals_are_kept_beside_the_library() {
    let stub = Stub::start(Box::new(ok_answer));
    let dir = temp_dir("queue");
    let lib = dir.join("lib");
    let mut s = Session::new();
    s.open_library(&lib, true).unwrap();
    s.execute("vera.describe.config", &json!({"endpoint": stub.base, "model": "stub-vlm"})).unwrap();
    let ids = distinct(&s);
    s.execute("photo.describe", &json!({"ids": [ids[0].0, ids[1].0]})).unwrap();
    drain(&mut s, 60);
    let file = lib.join(super::review::FILE_NAME);
    assert!(file.is_file(), "written when the run ends");
    // a photo deleted meanwhile loses its proposal when the library opens again
    s.execute("photo.deletePermanently", &json!({"ids": [ids[1].0]})).unwrap();
    s.close_library().unwrap();
    drop(s);
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    let l = proposals(&mut s);
    assert_eq!(l["total"], 1, "{l}");
    assert_eq!(l["proposals"][0]["photo"], ids[0].0);
    // a damaged file is a fresh start, not a crash
    s.close_library().unwrap();
    drop(s);
    std::fs::write(&file, b"{not json").unwrap();
    let mut s = Session::new();
    s.open_library(&lib, false).unwrap();
    assert_eq!(proposals(&mut s)["total"], 0);
    drop(s);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- zero-shot keywords with hand-made vectors

/// A matrix for the shipped vocabulary: each term along its own random direction (its phrasings a little apart).
fn fake_matrix(s: &Session) -> (Matrix, Vec<(String, Vec<f32>)>) {
    let vocab = s.vera.describe.keywords.vocab.clone();
    let mut terms = Vec::new();
    for f in &vocab.facets {
        for t in &f.terms {
            terms.push((t.keyword.clone(), random_vec(10_000 + terms.len() as u64)));
        }
    }
    let mut offsets = Vec::new();
    let mut n = 0;
    for f in &vocab.facets {
        offsets.push(n);
        n += f.terms.len();
    }
    let vectors: Vec<Vec<f32>> = vocab
        .prompts()
        .iter()
        .enumerate()
        .map(|(i, (f, t, _))| {
            let base = &terms[offsets[*f] + t].1;
            let noise = random_vec(90_000 + i as u64);
            unit(&base.iter().zip(&noise).map(|(a, b)| a + 0.1 * b).collect::<Vec<_>>())
        })
        .collect();
    (Matrix::from_vectors(super::lane::MODEL_ID, &vocab, DIM, &vectors).unwrap(), terms)
}

fn term<'a>(terms: &'a [(String, Vec<f32>)], k: &str) -> &'a [f32] {
    &terms.iter().find(|(t, _)| t == k).unwrap_or_else(|| panic!("no term {k}")).1
}

/// A photo vector leaning on two terms.
fn leaning(terms: &[(String, Vec<f32>)], a: &str, b: &str, seed: u64) -> Vec<f32> {
    let noise = random_vec(seed);
    unit(&term(terms, a).iter().zip(term(terms, b)).zip(&noise).map(|((x, y), z)| 0.6 * x + 0.5 * y + 0.4 * z).collect::<Vec<_>>())
}

#[test]
fn describe_photo_keywords_from_hand_made_vectors() {
    let mut s = Session::with_demo();
    let (m, terms) = fake_matrix(&s);
    s.vera.describe.keywords.matrix = Some(Arc::new(m));
    let ids = distinct(&s);
    let (a, b) = (ids[0], ids[1]);
    let ka = content_key(s.catalog.photo(a).unwrap());
    let kb = content_key(s.catalog.photo(b).unwrap());
    s.vera.lane.store.insert(ka, &leaning(&terms, "lake", "mountains", 7));
    s.vera.lane.store.insert(kb, &random_vec(8));
    let r = s.execute("photo.keywords", &json!({"id": a.0})).unwrap();
    let words: Vec<&str> = r["keywords"].as_array().unwrap().iter().map(|k| k["keyword"].as_str().unwrap()).collect();
    assert_eq!(words[..2], ["lake", "mountains"], "the two terms it leans on lead the subjects: {r}");
    assert!(words.len() <= 15);
    assert_eq!(r["keywords"][0]["facet"], "subjects");
    assert_eq!((r["calibrated"].as_bool(), r["id"].as_u64()), (Some(false), Some(a.0)), "under 200 vectors: raw cosines");
    assert!(r["ceiling"].as_f64().is_some_and(|c| c > 0.0 && c < 0.4), "{r}");
    assert!(r["keywords"][0]["cosine"].as_f64().unwrap() > r["keywords"][1]["cosine"].as_f64().unwrap());
    assert!(r["keywords"][0].get("z").is_none());
    // a vector near no term: what clears its ceiling is its own top tail, all of it far below the real terms
    let r = s.execute("photo.keywords", &json!({"id": b.0})).unwrap();
    assert!(r["keywords"].as_array().unwrap().iter().all(|k| k["cosine"].as_f64().unwrap() < 0.2), "{r}");
    assert_eq!(s.execute("photo.keywords", &json!({"id": a.0, "limit": 1})).unwrap()["keywords"].as_array().unwrap().len(), 1);
    // read-only
    assert!(!s.journal.iter().any(|(id, _)| id == "photo.keywords"));
    // hostile params and a photo without a vector
    for p in [json!({"id": "x"}), json!({"id": 999_999}), json!({"id": a.0, "limit": -1})] {
        assert!(s.execute("photo.keywords", &p).is_err(), "{p}");
    }
    let e = s.execute("photo.keywords", &json!({"id": ids[2].0})).unwrap_err().to_string();
    assert!(e.contains("embedding"), "{e}");
}

#[test]
fn describe_photo_keywords_calibrate_from_200_vectors() {
    let mut s = Session::with_demo();
    let (m, terms) = fake_matrix(&s);
    s.vera.describe.keywords.matrix = Some(Arc::new(m));
    let a = distinct(&s)[0];
    // a library of 250 pictures, all leaning on "sky" (a hub term); this photo leans on sky and lake
    for i in 0..250u64 {
        s.vera.lane.store.insert(format!("other-{i}"), &leaning(&terms, "sky", "clouds", 20_000 + i));
    }
    s.vera.lane.store.insert(content_key(s.catalog.photo(a).unwrap()), &leaning(&terms, "sky", "lake", 5));
    let r = s.execute("photo.keywords", &json!({"id": a.0})).unwrap();
    assert_eq!(r["calibrated"], true, "{r}");
    let rows = r["keywords"].as_array().unwrap();
    let best = rows.iter().max_by(|x, y| x["z"].as_f64().unwrap().total_cmp(&y["z"].as_f64().unwrap())).unwrap();
    assert_eq!(best["keyword"], "lake", "the hub term every picture shares does not win: {r}");
    assert!(s.vera.describe.keywords.calibration.as_ref().is_some_and(|c| c.sampled == 251));
}

#[test]
fn describe_keywords_only_needs_no_endpoint() {
    let mut s = Session::with_demo();
    let (m, terms) = fake_matrix(&s);
    s.vera.describe.keywords.matrix = Some(Arc::new(m));
    let ids = distinct(&s);
    let a = ids[0];
    s.vera.lane.store.insert(content_key(s.catalog.photo(a).unwrap()), &leaning(&terms, "desert", "dunes", 3));
    assert_eq!(s.execute("vera.status", &json!({})).unwrap()["lanes"]["describe"]["state"], "needs_endpoint");
    let r = s.execute("photo.describe", &json!({"ids": [a.0], "keywordsOnly": true})).unwrap();
    assert_eq!((r["queued"].as_u64(), r["keywordsOnly"].as_bool()), (Some(1), Some(true)), "{r}");
    assert!(r.get("host").is_none());
    drain(&mut s, 60);
    let l = proposals(&mut s);
    let p = &l["proposals"][0];
    let kw: Vec<&str> = p["fields"]["keywords"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
    assert_eq!(kw.first(), Some(&"dunes"), "subjects first, the best first: {l}");
    assert!(kw.contains(&"desert") && p["fields"].get("title").is_none(), "{l}");
    assert_eq!(p["source"], json!({"kind": "local", "model": super::lane::MODEL_ID}));
    s.execute("describe.accept", &json!({"id": a.0})).unwrap();
    let m = s.catalog.photo(a).unwrap().meta.clone();
    assert!(m.keywords.iter().any(|k| k == "dunes"));
    assert!(matches!(m.provenance.get("keywords"), Some(Provenance::Describe { model, .. }) if model == super::lane::MODEL_ID));
    assert!(!m.provenance.contains_key("title"), "only what was accepted");
    // a photo without a vector (and no model to make one) fails with a reason
    s.execute("photo.describe", &json!({"ids": [ids[1].0], "keywordsOnly": true})).unwrap();
    drain(&mut s, 60);
    let l = proposals(&mut s);
    assert!(l["failed"][0]["error"].as_str().unwrap().contains("SigLIP"), "{l}");
    // with no keyword vectors and no model, keywordsOnly is refused up front
    let mut bare = Session::with_demo();
    let e = bare.execute("photo.describe", &json!({"ids": [a.0], "keywordsOnly": true})).unwrap_err().to_string();
    assert!(e.contains("SigLIP 2"), "{e}");
}

/// The real vocabulary on the demo library: `LIGHTCRAFT_SIGLIP2_DIR=<dir> cargo test -p lightcraft-engine
/// describe_real -- --ignored --nocapture`. The library (and so the embedded vocabulary, a one-off of a few minutes) is
/// kept in `LIGHTCRAFT_VERA_TEST_LIB` (default: a folder in the temp dir) between runs; `LIGHTCRAFT_VERA_DUMP=<file>`
/// writes every scene's raw term cosines as JSON.
#[test]
#[ignore = "needs the SigLIP 2 checkpoint in LIGHTCRAFT_SIGLIP2_DIR"]
fn describe_real_keywords_recover_the_demo_scenes() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
        return;
    };
    let lib = std::env::var_os("LIGHTCRAFT_VERA_TEST_LIB").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("lc-vera-describe-real"));
    let mut s = Session::new();
    s.open_library(&lib, true).unwrap();
    s.vera_model_dir = Some(dir);
    let photos = s.catalog.photos().filter(|p| super::eligible(p)).count();
    let started = Instant::now();
    loop {
        let p = s.execute("vera.pump", &json!({"pace": "full"})).unwrap();
        assert_ne!(p["embedding"]["state"], "error", "{p}");
        if p["embedding"]["pending"] == 0 && p["embedding"]["embedded"].as_u64() == Some(photos as u64) {
            break;
        }
        assert!(started.elapsed().as_secs() < 1200);
        std::thread::sleep(Duration::from_millis(20));
    }
    eprintln!("embedded {photos} photos in {:.1} s", started.elapsed().as_secs_f64());
    let t = Instant::now();
    let first = distinct(&s)[0];
    loop {
        match s.execute("photo.keywords", &json!({"id": first.0})) {
            Ok(_) => break,
            Err(e) => assert!(e.to_string().contains("being embedded"), "{e}"),
        }
        assert!(t.elapsed().as_secs() < 1800);
        std::thread::sleep(Duration::from_millis(200));
    }
    eprintln!("the vocabulary ready in {:.1} s", t.elapsed().as_secs_f64());
    let scenes = lightcraft_scenes::demo_library();
    let vocab = s.vera.describe.keywords.vocab.clone();
    let matrix = s.vera.describe.keywords.matrix.clone().unwrap();
    let names: Vec<String> = vocab.facets.iter().flat_map(|f| f.terms.iter().map(|t| t.keyword.clone())).collect();
    let (mut found, mut total, mut scored) = (0usize, 0usize, 0usize);
    let mut dump = Vec::new();
    let rows: Vec<(PhotoId, String)> = distinct(&s).iter().map(|id| (*id, s.catalog.photo(*id).unwrap().meta.title.clone())).collect();
    for (id, title) in rows {
        let Some(scene) = scenes.iter().find(|sc| sc.name == title) else { continue };
        scored += 1;
        let v = s.vera.lane.store.get(&content_key(s.catalog.photo(id).unwrap())).unwrap().to_vec();
        let cos = super::keywords::term_cosines(&vocab, &matrix, &v).unwrap();
        let mut top: Vec<(f32, &str)> = cos.iter().copied().zip(names.iter().map(String::as_str)).collect();
        top.sort_by(|a, b| b.0.total_cmp(&a.0));
        dump.push(json!({"title": title, "own": scene.meta.keywords, "cosines": cos}));
        let r = s.execute("photo.keywords", &json!({"id": id.0, "limit": 10})).unwrap();
        let words: Vec<String> = r["keywords"].as_array().unwrap().iter().map(|k| k["keyword"].as_str().unwrap().to_string()).collect();
        let own: Vec<&str> = scene.meta.keywords.clone();
        let hit = own.iter().filter(|k| words.iter().any(|w| w == *k)).count();
        found += hit;
        total += own.len();
        let raw: Vec<String> = top.iter().take(6).map(|(c, t)| format!("{t} {c:.3}")).collect();
        eprintln!("{title:26} own {own:?} → {hit}/{} | proposed {words:?} (ceiling {}) | raw top {raw:?}", own.len(), r["ceiling"]);
        for banned in ["dog", "car", "people"] {
            assert!(!words.iter().any(|w| w == banned), "{title}: proposed {banned}: {words:?}");
        }
    }
    if let Some(path) = std::env::var_os("LIGHTCRAFT_VERA_DUMP") {
        std::fs::write(path, json!({"terms": names, "scenes": dump}).to_string()).unwrap();
    }
    let recall = found as f64 / total.max(1) as f64;
    eprintln!("recall@10 of the scenes' own keywords over {scored} scenes: {found}/{total} = {recall:.3}");
    assert_eq!(scored, 24);
    assert!(recall >= 0.5, "recall@10 {recall:.3}");
}
