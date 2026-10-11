//! Headless tests of VERA's Describe lane in the app (`panels::describe_review`, `panels::vera_settings`,
//! `docs/vera.md` → Phase 3): Settings ▸ Search ▸ Describe (the endpoint, the write-only token, the consent for another
//! computer), Photo ▸ Describe Photos… and Suggest Keywords, Window ▸ Review Descriptions (edit, untick, Accept,
//! Reject, Skip, Reject All) and the Info panel's "AI" tag. The endpoint is a stub OpenAI-compatible server on loopback
//! (as `crates/engine/src/vera/tests_describe.rs` has): nothing leaves the computer.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use lightcraft_catalog::{PhotoId, Provenance};
use serde_json::{Value, json};

use crate::headless::Headless;
use crate::state::{Dialog, RightPanel, ViewMode};
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(30);
const SETTLE: Duration = Duration::from_secs(120);
const SECRET: &str = "sk-describe-secret-4711";

// ---- a stub OpenAI-compatible server on loopback

/// Answers every `POST …/chat/completions` with the same proposal (after `slow` from the second request on).
struct Stub {
    base: String,
    requests: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
}

impl Stub {
    fn start() -> Stub {
        Stub::start_slow(Duration::ZERO)
    }

    fn start_slow(slow: Duration) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (count, stop2) = (requests.clone(), stop.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                if stop2.load(Ordering::Relaxed) {
                    return;
                }
                let Ok(mut sock) = conn else { continue };
                let count = count.clone();
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
                    assert!(head.starts_with("POST /v1/chat/completions "), "{head}");
                    if count.fetch_add(1, Ordering::SeqCst) > 0 {
                        std::thread::sleep(slow);
                    }
                    let reply = completion();
                    let out = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}", reply.len());
                    let _ = sock.write_all(out.as_bytes());
                });
            }
        });
        Stub { base, requests, stop }
    }

    fn host(&self) -> &str {
        self.base.trim_start_matches("http://")
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = std::net::TcpStream::connect(self.host());
    }
}

/// The proposal the stub writes, as a chat-completions answer.
fn completion() -> String {
    let content = json!({
        "title": "Lake Below the Peaks",
        "caption": "A still alpine lake at first light.",
        "alt": "A calm lake reflecting snowy mountains under a pale sky.",
        "keywords": ["lake", "mountains", "dawn"],
    });
    json!({"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": content.to_string()}}]})
        .to_string()
}

// ---- the app

fn app_with(png: bool) -> LightcraftApp {
    let services = if png {
        Services {
            write: Some(Box::new(lightcraft_engine::export::write_file)),
            png: Some(Box::new(|img: &lightcraft_raster::Rgba8| {
                lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default())
                    .unwrap_or_default()
            })),
            ..Default::default()
        }
    } else {
        Services { png: None, ..Default::default() }
    };
    let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), services);
    app.ui.view = ViewMode::PhotoGrid;
    app
}

/// The demo app once background fingerprinting is done (its activity row sits over the top of the grid until then).
fn demo() -> Headless {
    headless(app_with(false), [1400.0, 900.0])
}

fn headless(app: LightcraftApp, size: [f32; 2]) -> Headless {
    let mut h = Headless::new(app, size, 1.0);
    let done = h.step_until(SETTLE, |h| h.app.caches.vera_pending == 0 && h.app.caches.vera_digested > 0);
    assert!(done, "fingerprinting never finished: {} pending", h.app.caches.vera_pending);
    h.settle(SETTLE);
    h
}

/// A model folder of its own (as the desktop app has), so that a token has somewhere to go: its parent is where
/// `describe-remote.token` is kept. Nothing is downloaded into it.
fn model_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lightcraft-describe-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("models").join("siglip2-base-patch32-256")
}

fn ask(h: &mut Headless, method: &str, params: Value) -> Value {
    let r = h.request(method, params.clone(), T);
    assert_eq!(r["ok"], true, "{method} {params}: {r}");
    h.settle(SETTLE);
    r["result"].clone()
}

fn run(h: &mut Headless, command: &str, params: Value) -> Value {
    ask(h, "engine.execute", json!({"command": command, "params": params}))
}

fn click(h: &mut Headless, id: &str) {
    assert!(has_widget(h, id), "{id} is on screen: {:?}", h.app.widgets.iter().map(|(w, _)| w).collect::<Vec<_>>());
    ask(h, "ui.clickWidget", json!({"id": id}));
}

/// Click a text field, replace what it holds with `text`.
fn type_into(h: &mut Headless, field: &str, text: &str) {
    click(h, field);
    ask(h, "ui.key", json!({"key": "A", "cmd": true}));
    ask(h, "ui.text", json!({"text": text}));
}

fn key(h: &mut Headless, k: &str) {
    ask(h, "ui.key", json!({"key": k}));
}

fn painted(h: &Headless, text: &str) -> bool {
    h.painted_text().iter().any(|t| t.contains(text))
}

fn has_widget(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

fn toast(h: &Headless) -> String {
    h.app.ui.toast.as_ref().map(|t| t.0.clone()).unwrap_or_default()
}

/// Two demo photos that show different pictures, in id order.
fn two_photos(h: &Headless) -> (PhotoId, PhotoId) {
    let mut picked: Vec<&lightcraft_catalog::Photo> = Vec::new();
    for p in h.app.session.catalog.photos().filter(|p| p.copy_of.is_none()) {
        if picked.iter().all(|q| q.source != p.source) {
            picked.push(p);
        }
    }
    let mut ids: Vec<PhotoId> = picked.iter().map(|p| p.id).collect();
    ids.sort();
    (ids[0], ids[1])
}

fn proposals(h: &mut Headless) -> Value {
    h.app.session.execute("describe.list", &json!({})).unwrap()
}

fn review(h: &Headless) -> crate::state::DescribeReview {
    match &h.app.ui.dialog {
        Some(Dialog::ReviewDescriptions { review }) => review.clone(),
        other => panic!("the review is open: {other:?}"),
    }
}

/// Describe `ids` through the Photo menu and run the app until the proposals are in.
fn describe(h: &mut Headless, ids: &[PhotoId], want: u64) {
    run(h, "library.select", json!({"ids": ids.iter().map(|i| i.0).collect::<Vec<_>>()}));
    let r = ask(h, "ui.menu.invoke", json!({"id": "photo.describeSelected"}));
    assert_eq!(r["queued"].as_u64(), Some(ids.len() as u64), "{r}");
    let done = h.step_until(Duration::from_secs(90), |h| {
        let (pending, flight, proposals) = h.app.caches.vera_describe;
        pending == 0 && flight == 0 && proposals >= want
    });
    assert!(done, "describing never finished: {:?} {}", h.app.caches.vera_describe, h.app.session.execute("describe.list", &json!({})).unwrap());
    h.settle(SETTLE);
}

/// Settings ▸ Search ▸ Describe: a loopback endpoint and its model typed in and saved with Return, the token typed into
/// a password field that never paints it, then "Token set · Replace / Clear" (the stored token never shown); no consent
/// asked for this computer.
#[test]
fn describe_settings_set_the_endpoint_and_keep_the_token_write_only() {
    let stub = Stub::start();
    let mut h = demo();
    let dir = model_dir("settings");
    h.app.session.vera_model_dir = Some(dir.clone());
    run(&mut h, "app.settings", json!({"tab": "search"}));
    for text in ["Describe", "Needs an endpoint", "Endpoint", "Token", "llama.cpp, Ollama, vLLM or LM Studio", "Nothing is sent until you choose"] {
        assert!(painted(&h, text), "{text:?} is drawn: {:?}", h.painted_text());
    }
    assert!(!has_widget(&h, "check:describe.allowRemoteHost"), "no consent while nothing is set");
    type_into(&mut h, "field:describe.endpoint", &stub.base);
    assert!(!has_widget(&h, "check:describe.allowRemoteHost"), "this computer: no consent asked");
    type_into(&mut h, "field:describe.model", "stub-vlm");
    type_into(&mut h, "field:describe.token", SECRET);
    assert!(!h.painted_text().iter().any(|t| t.contains(SECRET) || t.contains("4711")), "the token is typed as dots: {:?}", h.painted_text());
    key(&mut h, "Enter");
    let c = run(&mut h, "vera.describe.config", json!({}));
    assert_eq!(
        (c["endpoint"].as_str(), c["model"].as_str(), c["tokenSet"].as_bool(), c["state"].as_str()),
        (Some(stub.base.as_str()), Some("stub-vlm"), Some(true), Some("live")),
        "{c}"
    );
    assert_eq!(c["allowRemoteHost"], false, "{c}");
    assert!(!c.to_string().contains(SECRET));
    let file = dir.parent().unwrap().join("describe-remote.token");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), SECRET, "kept in its private file");
    // the stored token is never painted: "Token set", Replace and Clear instead of the field
    assert!(painted(&h, "Ready") && painted(&h, "Token set"), "{:?}", h.painted_text());
    assert!(!h.painted_text().iter().any(|t| t.contains(SECRET)), "{:?}", h.painted_text());
    assert!(!has_widget(&h, "field:describe.token") && has_widget(&h, "button:describeTokenReplace"));
    // Replace shows an empty field (not the token)
    click(&mut h, "button:describeTokenReplace");
    assert!(has_widget(&h, "field:describe.token"));
    assert!(!h.painted_text().iter().any(|t| t.contains('•')), "an empty field: {:?}", h.painted_text());
    // Clear removes it; the lane goes on (a loopback endpoint needs none)
    h.app.ui.dialog = None;
    h.settle(SETTLE);
    run(&mut h, "app.settings", json!({"tab": "search"}));
    click(&mut h, "button:describeTokenClear");
    let c = run(&mut h, "vera.describe.config", json!({}));
    assert_eq!((c["tokenSet"].as_bool(), c["state"].as_str()), (Some(false), Some("live")), "{c}");
    assert!(!file.exists());
    assert!(has_widget(&h, "field:describe.token") && !painted(&h, "Token set"));
    // the lane's row says it is on
    assert!(h.painted_text().iter().filter(|t| t.as_str() == "On").count() >= 3, "{:?}", h.painted_text());
    let _ = std::fs::remove_dir_all(dir.parent().unwrap().parent().unwrap());
}

/// The consent appears only for another computer, in the PRD's words; without it the engine refuses and the section
/// says so. Ticked, a remote https endpoint is kept (and needs a token); plain http to another computer is refused
/// whatever the tick.
#[test]
fn describe_settings_ask_consent_only_for_another_computer() {
    let mut h = demo();
    h.app.session.vera_model_dir = Some(model_dir("consent"));
    run(&mut h, "app.settings", json!({"tab": "search"}));
    type_into(&mut h, "field:describe.endpoint", "https://vlm.example.com/v1");
    assert!(has_widget(&h, "check:describe.allowRemoteHost"), "another computer: the consent");
    assert!(painted(&h, crate::panels::vera_settings::REMOTE_CONSENT), "{:?}", h.painted_text());
    type_into(&mut h, "field:describe.model", "llava");
    click(&mut h, "button:describeSave");
    assert!(painted(&h, "vlm.example.com is another computer: tick the box above"), "{:?}", h.painted_text());
    assert!(run(&mut h, "vera.describe.config", json!({}))["endpoint"].is_null(), "refused: nothing kept");
    // ticked: kept, and a token is what it waits for
    click(&mut h, "check:describe.allowRemoteHost");
    click(&mut h, "button:describeSave");
    let c = run(&mut h, "vera.describe.config", json!({}));
    assert_eq!(
        (c["endpoint"].as_str(), c["allowRemoteHost"].as_bool(), c["state"].as_str()),
        (Some("https://vlm.example.com/v1"), Some(true), Some("needs_token")),
        "{c}"
    );
    assert!(painted(&h, "Needs a token") && !has_widget(&h, "vera:describe:error"), "{:?}", h.painted_text());
    // plain http to another computer: refused, ticked or not
    type_into(&mut h, "field:describe.endpoint", "http://192.168.1.5:8080");
    assert!(has_widget(&h, "check:describe.allowRemoteHost"));
    click(&mut h, "button:describeSave");
    assert!(painted(&h, "https://"), "the engine's reason: {:?}", h.painted_text());
    assert_eq!(run(&mut h, "vera.describe.config", json!({}))["endpoint"], "https://vlm.example.com/v1", "unchanged");
    // back to this computer: no consent asked, and it is taken back with the endpoint
    type_into(&mut h, "field:describe.endpoint", "http://localhost:11434");
    assert!(!has_widget(&h, "check:describe.allowRemoteHost"));
    key(&mut h, "Enter");
    let c = run(&mut h, "vera.describe.config", json!({}));
    assert_eq!(
        (c["endpoint"].as_str(), c["allowRemoteHost"].as_bool(), c["state"].as_str()),
        (Some("http://localhost:11434"), Some(false), Some("live")),
        "{c}"
    );
}

/// The loopback test is what decides when the consent shows.
#[test]
fn describe_loopback_hosts_are_told_apart() {
    use crate::panels::vera_settings::{endpoint_host, is_loopback};
    for e in ["http://127.0.0.1:11434", "http://localhost:8080/v1", "http://[::1]:8080", "127.0.0.2:1", "", "http://LOCALHOST"] {
        assert!(is_loopback(e), "{e}");
    }
    for e in
        ["https://vlm.example.com", "http://192.168.1.5:8080", "https://user@host.lan/v1", "http://[2001:db8::1]:80", "http://127.0.0.1.example.com"]
    {
        assert!(!is_loopback(e), "{e}");
    }
    assert_eq!(endpoint_host("https://user@Host.Lan:8443/v1?x"), "host.lan");
}

/// Photo ▸ Describe Photos… on two demo photos against the stub; the run's end offers the review. Window ▸ Review
/// Descriptions shows the first proposal (photo, source line, each field beside the photo's own value): the title is
/// edited, alt text unticked and a keyword dropped before Accept, which writes exactly that with Describe's provenance
/// in one undo step. Backspace rejects the second (nothing written). Skip and Reject All on a second run; the Info
/// panel then marks the accepted fields "AI".
#[test]
fn describe_photos_review_accept_and_reject() {
    let stub = Stub::start();
    let mut h = demo();
    run(&mut h, "vera.describe.config", json!({"endpoint": stub.base, "model": "stub-vlm"}));
    let (a, b) = two_photos(&h);
    let before_a = h.app.session.catalog.photo(a).unwrap().meta.clone();
    let before_b = h.app.session.catalog.photo(b).unwrap().meta.clone();
    // the menu items, enabled with a selection; the context menu has Describe too
    let photo = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "Photo").expect("Photo menu").1;
    for (id, label) in [("photo.describeSelected", "Describe Photos…"), ("photo.keywordsSelected", "Suggest Keywords")] {
        let item = photo.iter().find_map(|n| match n {
            crate::menubar::MenuNode::Item { id: i, label, enabled, .. } if i == id => Some((label.clone(), *enabled)),
            _ => None,
        });
        assert_eq!(item, Some((label.to_string(), true)), "{id}");
        assert!(crate::menus::UI_COMMANDS.iter().any(|c| c.0 == id));
    }
    let window = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "Window").expect("Window menu").1;
    assert!(window.iter().any(|n| matches!(n, crate::menubar::MenuNode::Item { id, .. } if id == "view.reviewDescriptions")));

    describe(&mut h, &[a, b], 2);
    assert_eq!(stub.requests.load(Ordering::SeqCst), 2, "one request a photo");
    assert_eq!(proposals(&mut h)["total"], 2);
    assert_eq!(h.app.session.catalog.photo(a).unwrap().meta, before_a, "nothing written yet");
    // the end of the run says so, with a button to the review
    assert_eq!(toast(&h), "2 descriptions wait for your review");
    assert!(has_widget(&h, "button:toastAction") && painted(&h, "Review"), "{:?}", h.painted_text());
    click(&mut h, "button:toastAction");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::ReviewDescriptions { .. })), "{:?}", h.app.ui.dialog);
    h.settle(SETTLE);
    assert_eq!(review(&h).photo, Some(a.0), "the first proposal");
    for text in [
        "Review Descriptions",
        &format!("Proposed by stub-vlm at {}", stub.host()),
        "Now",
        "Proposed",
        "Lake Below the Peaks",
        "A still alpine lake at first light.",
        "2 to review",
        "This session: 0 accepted · 0 rejected · 0 being described · 2 to review",
    ] {
        assert!(painted(&h, text), "{text:?} is drawn: {:?}", h.painted_text());
    }
    let name = h.app.session.catalog.photo(a).unwrap().file_name.clone();
    assert!(painted(&h, &name));
    for f in ["title", "caption", "alt_text", "keywords"] {
        assert!(has_widget(&h, &format!("check:describe.{f}")), "{f} has its tick");
    }
    assert!(has_widget(&h, "describe:dropKeyword:dawn") && has_widget(&h, "field:describe.addKeyword"));
    // edit the title, untick alt text, drop a keyword and add one
    type_into(&mut h, "field:describe.title", "My Lake");
    key(&mut h, "Enter");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::ReviewDescriptions { .. })), "Return in a field ends the edit, not the review");
    assert_eq!(proposals(&mut h)["total"], 2, "and accepts nothing");
    click(&mut h, "check:describe.alt_text");
    click(&mut h, "describe:dropKeyword:dawn");
    type_into(&mut h, "field:describe.addKeyword", "tarn");
    key(&mut h, "Enter");
    let r = review(&h);
    assert_eq!((r.title.as_str(), r.keywords.clone()), ("My Lake", vec!["lake".to_string(), "mountains".into(), "tarn".into()]));
    let undo = h.app.session.undo.len();
    click(&mut h, "button:describeAccept");
    let m = h.app.session.catalog.photo(a).unwrap().meta.clone();
    assert_eq!(m.title, "My Lake");
    assert_eq!(m.caption, "A still alpine lake at first light.");
    assert_eq!(m.alt_text, before_a.alt_text, "unticked: left alone");
    for k in ["lake", "mountains", "tarn"] {
        assert!(m.keywords.iter().any(|x| x.eq_ignore_ascii_case(k)), "{k}: {:?}", m.keywords);
    }
    assert!(!m.keywords.iter().any(|x| x == "dawn") || before_a.keywords.iter().any(|x| x == "dawn"), "dropped: {:?}", m.keywords);
    for k in &before_a.keywords {
        assert!(m.keywords.contains(k), "added to the photo's own: {:?}", m.keywords);
    }
    for f in ["title", "caption", "keywords"] {
        assert!(matches!(m.provenance.get(f), Some(Provenance::Describe { model, .. }) if model == "stub-vlm"), "{f}: {:?}", m.provenance);
    }
    assert!(!m.provenance.contains_key("alt_text"), "{:?}", m.provenance);
    assert_eq!(h.app.session.undo.len(), undo + 1, "one undo step");
    assert_eq!(h.app.session.undo.last().unwrap().label, "Accept Description");
    // the next proposal is the other photo's; Backspace rejects it
    assert_eq!(review(&h).photo, Some(b.0), "{} {:?}", proposals(&mut h), h.painted_text());
    assert!(painted(&h, "This session: 1 accepted · 0 rejected"), "{:?}", h.painted_text());
    key(&mut h, "Backspace");
    assert_eq!(proposals(&mut h)["total"], 0, "gone");
    assert_eq!(h.app.session.catalog.photo(b).unwrap().meta, before_b, "a rejected proposal writes nothing");
    assert!(h.app.session.catalog.photo(b).is_some(), "and Delete never reaches the photo");
    assert!(painted(&h, "This session: 1 accepted · 1 rejected · 0 being described · 0 to review"), "{:?}", h.painted_text());
    assert!(has_widget(&h, "describe:empty"));

    // a second run: → skips to the next, Reject All asks first
    describe(&mut h, &[a, b], 2);
    assert_eq!(review(&h).photo, Some(a.0), "{} {:?}", proposals(&mut h), h.painted_text());
    key(&mut h, "Right");
    assert_eq!(review(&h).photo, Some(b.0), "skipped");
    key(&mut h, "Right");
    assert_eq!(review(&h).photo, Some(a.0), "round again");
    click(&mut h, "button:describeRejectAll");
    assert!(painted(&h, "Reject all 2 proposals? Nothing is written to the photos."), "{:?}", h.painted_text());
    click(&mut h, "button:describeRejectAllCancel");
    assert_eq!(proposals(&mut h)["total"], 2, "kept");
    click(&mut h, "button:describeRejectAll");
    click(&mut h, "button:describeRejectAllConfirm");
    assert_eq!(proposals(&mut h)["total"], 0);
    assert_eq!(review(&h).rejected, 3);
    assert_eq!(h.app.session.catalog.photo(a).unwrap().meta.title, "My Lake", "Reject All writes nothing");
    click(&mut h, "button:dialogOk");
    assert_eq!(h.app.ui.dialog, None, "Close");

    // the Info panel marks Describe's fields with "AI" (the model on hover); the field left alone has none
    run(&mut h, "library.select", json!({"ids": [a.0]}));
    h.app.ui.right = RightPanel::Info;
    h.settle(SETTLE);
    for f in ["title", "caption"] {
        assert!(has_widget(&h, &format!("ai:{f}")), "{f} is marked");
    }
    assert!(!has_widget(&h, "ai:altText") && !has_widget(&h, "ai:copyright"));
    assert!(painted(&h, "AI"));
    // typed over, the words are the user's
    type_into(&mut h, "field:caption", "Mine.");
    // leaving the field keeps the text
    click(&mut h, "field:copyright");
    let m = h.app.session.catalog.photo(a).unwrap().meta.clone();
    assert_eq!((m.caption.as_str(), m.provenance.get("caption")), ("Mine.", Some(&Provenance::User)), "{:?}", m.provenance);
    assert!(!has_widget(&h, "ai:caption") && has_widget(&h, "ai:title"));
    h.app.ui.right = RightPanel::Keywords;
    h.settle(SETTLE);
    assert!(has_widget(&h, "ai:keywords"), "the keywords too");
}

/// While a run brings proposals in, the activity stack's Describe row ("Describing photos", the host as its detail)
/// offers those already in for review; the review shows them while the rest are still being described.
#[test]
fn describe_activity_row_offers_the_review() {
    // the first answer at once, the second after a while: the run is still going with one proposal in
    let stub = Stub::start_slow(Duration::from_secs(20));
    let mut h = demo();
    run(&mut h, "vera.describe.config", json!({"endpoint": stub.base, "model": "stub-vlm"}));
    let (a, b) = two_photos(&h);
    run(&mut h, "library.select", json!({"ids": [a.0, b.0]}));
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.describeSelected"}));
    assert_eq!(r["queued"], 2, "{r}");
    assert_eq!(toast(&h), format!("Describing 2 photos with {}: each description waits for your review", stub.host()));
    let row = h.app.session.activity.list().into_iter().find(|t| t.kind == "describe").expect("the row");
    assert_eq!(row.label, "Describing photos");
    assert!(row.detail.contains(stub.host()), "{row:?}");
    let shown = h.step_until(T, |h| h.app.widgets.iter().any(|(w, _)| w == "activity:reviewDescriptions"));
    assert!(shown, "the row offers the review: {:?}", h.app.caches.vera_describe);
    // under load the second answer can land before this frame paints: either count is the row doing its job
    assert!(painted(&h, "1 to review") || painted(&h, "2 to review"), "{:?}", h.painted_text());
    click(&mut h, "activity:reviewDescriptions");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::ReviewDescriptions { .. })));
    assert!(painted(&h, "1 being described") || painted(&h, "0 being described"), "{:?}", h.painted_text());
    // the second comes in while the review is open
    let done = h.step_until(Duration::from_secs(90), |h| h.app.caches.vera_describe == (0, 0, 2));
    assert!(done, "{:?}", h.app.caches.vera_describe);
    h.settle(SETTLE);
    assert!(painted(&h, "0 being described · 2 to review"), "{:?}", h.painted_text());
    assert!(h.app.ui.toast.as_ref().is_none_or(|t| !t.0.contains("wait for your review")), "no toast over the open review");
}

/// Describe without an endpoint: nothing is queued, the toast says what is missing and opens Settings ▸ Search.
#[test]
fn describe_without_an_endpoint_says_where_to_set_one() {
    let mut h = demo();
    let r = h.request("ui.menu.invoke", json!({"id": "photo.describeSelected"}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["error"].as_str().unwrap_or("").contains("vera.describe.config"), "agents get the engine's error: {r}");
    assert_eq!(toast(&h), "Describing photos needs an endpoint: a llama.cpp, Ollama, vLLM or LM Studio server, set in Settings → Search.");
    assert!(h.app.session.activity.list().iter().all(|t| t.kind != "describe"), "nothing runs");
    assert!(has_widget(&h, "button:toastAction") && painted(&h, "Open Settings → Search"));
    click(&mut h, "button:toastAction");
    assert_eq!(h.app.ui.dialog, Some(Dialog::Settings { tab: "search".into() }));
    // nothing selected: disabled
    h.app.ui.dialog = None;
    h.app.session.selection = Default::default();
    assert!(!crate::menus::ui_enabled(&h.app, "photo.describeSelected"));
    // the review with nothing in it says how to get something
    run(&mut h, "app.settings", json!({"tab": "general"}));
    ask(&mut h, "ui.menu.invoke", json!({"id": "view.reviewDescriptions"}));
    assert!(painted(&h, "Nothing to review."), "{:?}", h.painted_text());
}

/// Suggest Keywords without the SigLIP 2 model: the toast says it needs the model and opens Settings ▸ Search.
#[test]
fn describe_suggest_keywords_without_the_model_says_so() {
    let mut h = demo();
    h.app.session.vera_model_dir = Some(model_dir("keywords"));
    let r = h.request("ui.menu.invoke", json!({"id": "photo.keywordsSelected"}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["error"].as_str().unwrap_or("").contains("SigLIP 2"), "{r}");
    assert_eq!(toast(&h), "Suggesting keywords needs the SigLIP 2 model: a one-time download, in Settings → Search.");
    assert!(h.app.ui.toast_action.is_some());
    // nowhere to keep a model: the engine's reason, nothing to open
    h.app.session.vera_model_dir = None;
    let r = h.request("ui.menu.invoke", json!({"id": "photo.keywordsSelected"}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false);
    assert!(toast(&h).contains("nowhere to keep it"), "{}", toast(&h));
    assert!(h.app.ui.toast_action.is_none());
}

/// Headless screenshots of Settings ▸ Search ▸ Describe (a loopback endpoint with a token; another computer with the
/// consent) and the review with a proposal open: written only when `LIGHTCRAFT_VERA_SHOTS` names a folder.
#[test]
fn describe_screenshots_of_the_settings_and_the_review() {
    let Some(out) = std::env::var_os("LIGHTCRAFT_VERA_SHOTS").map(PathBuf::from) else {
        eprintln!("skipped: set LIGHTCRAFT_VERA_SHOTS to a folder for the screenshots");
        return;
    };
    let stub = Stub::start();
    let mut app = app_with(true);
    let dir = model_dir("shots");
    app.session.vera_model_dir = Some(dir.clone());
    let mut h = headless(app, [1600.0, 1000.0]);
    let shot = |h: &mut Headless, name: &str| {
        let path = out.join(name);
        let r = h.request("ui.screenshot", json!({"path": path.to_string_lossy(), "headless": true}), SETTLE);
        assert_eq!(r["ok"], true, "{r}");
        eprintln!("wrote {}", path.display());
    };
    run(&mut h, "app.settings", json!({"tab": "search"}));
    type_into(&mut h, "field:describe.endpoint", &stub.base);
    type_into(&mut h, "field:describe.model", "qwen2.5vl");
    type_into(&mut h, "field:describe.token", SECRET);
    click(&mut h, "button:describeSave");
    shot(&mut h, "describe-settings.png");
    type_into(&mut h, "field:describe.endpoint", "https://vlm.example.com/v1");
    click(&mut h, "button:describeSave");
    shot(&mut h, "describe-settings-remote.png");
    h.app.ui.dialog = None;
    run(&mut h, "vera.describe.config", json!({"endpoint": stub.base, "model": "qwen2.5vl"}));
    let (a, b) = two_photos(&h);
    describe(&mut h, &[a, b], 2);
    ask(&mut h, "ui.menu.invoke", json!({"id": "view.reviewDescriptions"}));
    click(&mut h, "check:describe.alt_text");
    type_into(&mut h, "field:describe.title", "Lake Below the Peaks at Dawn");
    key(&mut h, "Enter");
    h.settle(SETTLE);
    shot(&mut h, "describe-review.png");
    click(&mut h, "button:describeAccept");
    click(&mut h, "button:dialogOk");
    run(&mut h, "library.select", json!({"ids": [a.0]}));
    h.app.ui.right = RightPanel::Info;
    h.settle(SETTLE);
    shot(&mut h, "describe-info-ai-tag.png");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap().parent().unwrap());
}
