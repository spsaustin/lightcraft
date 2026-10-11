//! Headless tests of VER Records in the app (`panels::ver`, `docs/ver.md`): Settings ▸ Search ▸ VER Records
//! (`ver.status`, `ver.config`), Photo ▸ Show VER Record… (made off the UI thread), File ▸ Read VER Record…, File ▸
//! Write VER Records…, the export dialog's "Write VER Records" and its note, and the duplicates view's "Same pixels".
//! The demo library has no embedding model, so a photo's Record ends in `needs_model`; the conformant dialog is drawn
//! from a Record of the standard's conformance suite.

use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lightcraft_catalog::PhotoId;
use serde_json::{Value, json};

use crate::headless::Headless;
use crate::pick::{PickKind, PickRequest};
use crate::state::{Dialog, ViewMode};
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(30);
const SETTLE: Duration = Duration::from_secs(120);

/// A folder of the test's own in the temp directory, removed when it goes.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("lightcraft-ver-ui-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// What the file dialogs were asked, and the answers they give (in order; none left: cancelled).
#[derive(Clone, Default)]
struct Dialogs {
    asked: Arc<Mutex<Vec<PickRequest>>>,
    answers: Arc<Mutex<Vec<String>>>,
}

impl Dialogs {
    fn answer(&self, path: &str) {
        self.answers.lock().unwrap().push(path.to_string());
    }

    fn last(&self) -> PickRequest {
        self.asked.lock().unwrap().last().cloned().expect("a file dialog")
    }
}

fn picture(w: usize, h: usize, seed: u8) -> lightcraft_raster::Rgba8 {
    let data = (0..w * h).map(|i| [((i % w) * 255 / w) as u8, ((i / w) * 255 / h) as u8, seed.wrapping_add((i % 7) as u8), 255]).collect();
    lightcraft_raster::Rgba8 { width: w, height: h, data }
}

fn write_jpeg(path: &str, w: usize, h: usize) {
    let img = picture(w, h, 40);
    let bytes = lightcraft_codecs::encode_jpeg(
        &lightcraft_codecs::EncodeImage::rgba8(&img),
        90,
        lightcraft_codecs::ChromaSubsampling::S420,
        &lightcraft_codecs::EncodeMeta::default(),
    )
    .unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// A PNG of the same pixels whatever `ppi` says: a different file (its `pHYs` chunk), the same rendered pixels.
fn write_png(path: &str, ppi: Option<u16>) {
    let img = picture(48, 32, 9);
    let meta = lightcraft_codecs::EncodeMeta { ppi, ..Default::default() };
    std::fs::write(path, lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &meta).unwrap()).unwrap();
}

/// The demo library, with the files in `import` added, file dialogs that answer from [`Dialogs`], a writer for
/// background exports and a PNG encoder for screenshots. Waits for background fingerprinting to finish.
fn app(import: &[String]) -> (Headless, Dialogs) {
    app_sized(import, [1400.0, 900.0])
}

fn app_sized(import: &[String], size: [f32; 2]) -> (Headless, Dialogs) {
    let dialogs = Dialogs::default();
    let (asked, answers) = (dialogs.asked.clone(), dialogs.answers.clone());
    let services = Services {
        picker: Some(Box::new(move |req| {
            asked.lock().unwrap().push(req);
            let (tx, rx) = channel();
            let mut a = answers.lock().unwrap();
            let answer = if a.is_empty() { Vec::new() } else { vec![a.remove(0)] };
            tx.send(answer).unwrap();
            Ok(rx)
        })),
        write: Some(Box::new(lightcraft_engine::export::write_file)),
        write_shared: Some(Arc::new(lightcraft_engine::export::write_file)),
        png: Some(Box::new(|img: &lightcraft_raster::Rgba8| {
            lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default()).unwrap_or_default()
        })),
        ..Default::default()
    };
    let mut session = lightcraft_engine::Session::with_demo().with_fs();
    if !import.is_empty() {
        session.execute("library.import", &json!({"paths": import})).unwrap();
    }
    let mut app = LightcraftApp::new(session, services);
    app.ui.view = ViewMode::PhotoGrid;
    let mut h = Headless::new(app, size, 1.0);
    let done = h.step_until(SETTLE, |h| h.app.caches.vera_pending == 0 && h.app.caches.vera_digested > 0);
    assert!(done, "fingerprinting never finished: {} pending", h.app.caches.vera_pending);
    h.settle(SETTLE);
    (h, dialogs)
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

fn has_widget(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

fn click(h: &mut Headless, id: &str) {
    assert!(has_widget(h, id), "{id} is on screen: {:?}", h.app.widgets.iter().map(|(w, _)| w).collect::<Vec<_>>());
    ask(h, "ui.clickWidget", json!({"id": id}));
}

fn type_into(h: &mut Headless, field: &str, text: &str) {
    click(h, field);
    ask(h, "ui.key", json!({"key": "A", "cmd": true}));
    ask(h, "ui.text", json!({"text": text}));
}

fn painted(h: &Headless, text: &str) -> bool {
    h.painted_text().iter().any(|t| t.contains(text))
}

fn toast(h: &Headless) -> String {
    h.app.ui.toast.as_ref().map(|t| t.0.clone()).unwrap_or_default()
}

/// Write a headless screenshot when `LIGHTCRAFT_VER_SHOTS` names a folder.
fn shot(h: &mut Headless, name: &str) {
    let Some(out) = std::env::var_os("LIGHTCRAFT_VER_SHOTS").map(PathBuf::from) else { return };
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join(name);
    let r = h.request("ui.screenshot", json!({"path": path.to_string_lossy(), "headless": true}), SETTLE);
    assert_eq!(r["ok"], true, "{r}");
    eprintln!("wrote {}", path.display());
}

fn photo_named(h: &Headless, name: &str) -> PhotoId {
    h.app.session.catalog.photos().find(|p| p.file_name == name).map(|p| p.id).unwrap_or_else(|| panic!("no photo {name}"))
}

/// A Record of the standard's conformance suite (VER 1.1, L3).
fn conformance_record(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ver/standard/conformance").join(name)
}

/// An unencrypted Ed25519 PKCS#8 PEM (RFC 8410) for seed `seed`.
fn ed25519_pem(seed: u8) -> String {
    let mut der = vec![0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20];
    der.extend((0..32u8).map(|i| i.wrapping_mul(7).wrapping_add(seed)));
    const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in der.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    format!("-----BEGIN PRIVATE KEY-----\n{out}\n-----END PRIVATE KEY-----\n")
}

/// Settings ▸ Search ▸ VER Records says why Records cannot be made here (no model: the demo has none) with the way to
/// get it, and its fields round-trip through `ver.config`: the authority (a bad one is refused in the section), the
/// export sidecars switch, the default schema and the signing key (its key id shown, then cleared). The fingerprint
/// counts say how many are pixel-hashed and provisional.
#[test]
fn ver_settings_round_trip_through_ver_config_and_show_the_status() {
    let dir = Scratch::new("settings");
    // tall enough for all of Settings ▸ Search without scrolling
    let (mut h, dialogs) = app_sized(&[], [1400.0, 1400.0]);
    let config = |h: &mut Headless| run(h, "ver.config", json!({}));
    assert_eq!(config(&mut h)["authority"], "org.verstandard");
    run(&mut h, "app.settings", json!({"tab": "search"}));
    assert!(painted(&h, "VER Records"), "{:?}", h.painted_text());
    assert!(painted(&h, "Needs the model: Records carry an embedding"), "{:?}", h.painted_text());
    assert!(has_widget(&h, "ver:status"));
    let counts = h.painted_text().into_iter().find(|t| t.contains("photos fingerprinted")).unwrap_or_default();
    assert!(counts.contains("pixel-hashed 0 · provisional"), "the demo's scenes have no VER path: {counts}");
    shot(&mut h, "ver-settings.png");

    // the authority: Return applies it; a bad one is refused and says why
    type_into(&mut h, "field:ver.authority", "com.example.photos");
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    let c = config(&mut h);
    assert_eq!(c["authority"], "com.example.photos", "{c} {:?} {:?}", h.painted_text(), h.app.widgets.iter().map(|(w, _)| w).collect::<Vec<_>>());
    assert_eq!(run(&mut h, "ver.status", json!({}))["authority"], "com.example.photos");
    type_into(&mut h, "field:ver.authority", "Not A Domain");
    click(&mut h, "button:verApply");
    assert!(has_widget(&h, "ver:error"), "the refusal is shown");
    assert!(painted(&h, "reverse-DNS"), "{:?}", h.painted_text());
    assert_eq!(config(&mut h)["authority"], "com.example.photos", "nothing changed");
    // emptied: back to the default
    click(&mut h, "field:ver.authority");
    ask(&mut h, "ui.key", json!({"key": "A", "cmd": true}));
    ask(&mut h, "ui.key", json!({"key": "Backspace"}));
    click(&mut h, "button:verApply");
    assert_eq!(config(&mut h)["authority"], "org.verstandard");
    assert!(!has_widget(&h, "ver:error"));

    // the export sidecars switch and the default schema apply at once
    assert_eq!(config(&mut h)["exportSidecars"], true);
    click(&mut h, "check:ver.exportSidecars");
    assert_eq!(config(&mut h)["exportSidecars"], false);
    click(&mut h, "check:ver.exportSidecars");
    assert_eq!(config(&mut h)["exportSidecars"], true);
    click(&mut h, "button:verSchema-1");
    assert_eq!(config(&mut h)["defaultSchema"], "1.0.1");
    click(&mut h, "button:verSchema-0");
    assert_eq!(config(&mut h)["defaultSchema"], "1.1-draft");

    // the signing key: a file that is no key is refused; a key is checked, kept and named by its key id; Clear
    std::fs::write(dir.path("not-a-key.pem"), "hello").unwrap();
    dialogs.answer(&dir.path("not-a-key.pem"));
    click(&mut h, "button:verKeyChoose");
    let req = dialogs.last();
    assert_eq!((req.kind, req.filter.as_ref().map(|f| f.1)), (PickKind::File, Some(&["pem", "key"][..])));
    assert!(toast(&h).contains("PEM"), "{}", toast(&h));
    assert_eq!(run(&mut h, "ver.status", json!({}))["signing"]["enabled"], false);
    std::fs::write(dir.path("key.pem"), ed25519_pem(3)).unwrap();
    dialogs.answer(&dir.path("key.pem"));
    click(&mut h, "button:verKeyChoose");
    let status = run(&mut h, "ver.status", json!({}));
    let key = status["signing"]["keyId"].as_str().expect("a key id").to_string();
    assert!(key.starts_with("ed25519:"), "{status}");
    assert!(painted(&h, &key), "the key id is shown");
    click(&mut h, "button:verKeyClear");
    assert_eq!(run(&mut h, "ver.status", json!({}))["signing"]["enabled"], false);
    assert!(painted(&h, "None: Records are unsigned"));
}

/// Photo ▸ Show VER Record… on a JPEG: the dialog opens at once saying "Writing the Record…" while the engine's VER
/// thread works (frames keep coming), then says in the app's words that a Record needs the model, with the way to
/// Settings ▸ Search. A demo scene has no file and says so. The menu item needs an active photo.
#[test]
fn show_ver_record_says_the_jpeg_needs_the_model_without_blocking_frames() {
    let dir = Scratch::new("show");
    let jpeg = dir.path("harbour.jpg");
    write_jpeg(&jpeg, 1600, 1200);
    let (mut h, _) = app(std::slice::from_ref(&jpeg));
    let photo_menu = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "Photo").expect("Photo menu").1;
    let item = photo_menu.iter().find_map(|n| match n {
        crate::menubar::MenuNode::Item { id, label, enabled, .. } if id == "photo.showVerRecord" => Some((label.clone(), *enabled)),
        _ => None,
    });
    assert_eq!(item, Some(("Show VER Record…".to_string(), true)));
    let selection = h.app.session.selection.clone();
    h.app.session.selection = Default::default();
    assert!(!crate::menus::ui_enabled(&h.app, "photo.showVerRecord"), "no active photo: disabled");
    h.app.session.selection = selection;

    let id = photo_named(&h, "harbour.jpg");
    run(&mut h, "library.select", json!({"ids": [id.0]}));
    let t0 = Instant::now();
    let r = h.app.run("photo.showVerRecord", json!({})).unwrap();
    assert!(t0.elapsed() < Duration::from_secs(2), "the command returns at once");
    assert_eq!(r["background"], true, "{r}");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::VerRecord { photo }) if photo == id.0));
    assert!(h.app.caches.ver.view.as_ref().is_some_and(|v| v.running()));
    // frames go on while it is written, each of them short
    let (mut running_frames, mut longest) = (0, Duration::ZERO);
    let deadline = Instant::now() + SETTLE;
    while h.app.caches.ver.view.as_ref().is_some_and(|v| v.running()) {
        let t = Instant::now();
        h.step();
        longest = longest.max(t.elapsed());
        if painted(&h, "Writing the Record…") {
            running_frames += 1;
        }
        assert!(Instant::now() < deadline, "the Record never came");
    }
    assert!(running_frames >= 1, "the dialog said it was writing the Record");
    assert!(longest < Duration::from_secs(2), "no frame waited for the Record: {longest:?}");
    h.settle(SETTLE);
    let view = h.app.caches.ver.view.clone().unwrap();
    assert_eq!(view.error.as_ref().map(|e| e.0.as_str()), Some("needs_model"), "{view:?}");
    assert!(painted(&h, "needs the search-by-meaning model"), "{:?}", h.painted_text());
    assert!(has_widget(&h, "verRecord:error") && has_widget(&h, "button:verOpenSearch"));
    assert!(h.app.caches.ver.record_job.is_none(), "the job is over");
    shot(&mut h, "ver-show-record-needs-model.png");
    click(&mut h, "button:verOpenSearch");
    assert_eq!(h.app.ui.dialog, Some(Dialog::Settings { tab: "search".into() }));

    // a demo scene: no file, no Record, said at once
    let demo = h.app.session.catalog.photos().find(|p| matches!(p.source, lightcraft_catalog::Source::Demo { .. })).map(|p| p.id).unwrap();
    ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showVerRecord", "params": {"id": demo.0}}));
    let view = h.app.caches.ver.view.clone().unwrap();
    assert_eq!(view.error.as_ref().map(|e| e.0.as_str()), Some("no_asset"), "{view:?}");
    assert!(painted(&h, "A generated demo photo: it has no file, so it has no Record."));
}

/// The Record dialog with a Record (one of the standard's conformance Records, in `ver.record`'s shape): its badge,
/// level and schema, the identity block with Copy buttons, the space, the issues and the JSON, and Save… writing it
/// where the save dialog says.
#[test]
fn the_ver_record_dialog_shows_a_record_and_saves_its_json() {
    let dir = Scratch::new("dialog");
    let (mut h, dialogs) = app(&[]);
    let text = std::fs::read_to_string(conformance_record("valid/l3-full-1-1.json")).unwrap();
    let record: Value = serde_json::from_str(&text).unwrap();
    let id = h.app.session.active().unwrap();
    h.app.caches.ver.view = Some(crate::panels::ver::RecordView { photo: id.0, name: "harbour.jpg".into(), ..Default::default() });
    let issues = json!([{"code": "VER404", "severity": "warning", "pointer": "/spaces/0", "message": "the space declares no reference set"}]);
    crate::panels::ver::record_done(
        &mut h.app,
        id.0,
        json!({"record": record, "issues": issues, "conformant": true, "level": "L3", "notes": [], "schema": "1.1-draft"}),
    );
    h.app.ui.dialog = Some(Dialog::VerRecord { photo: id.0 });
    h.settle(SETTLE);
    for text in [
        "Conformant",
        "Level L3",
        "VER 1.1 (draft)",
        "VER404",
        "/spaces/0",
        "6f9902b45f5f277b6bb3a19604e008c7964234259b4f747611e0dca5c2d27cb7",
        "128 × 48",
        "a61964b34c4db96e",
    ] {
        assert!(painted(&h, text), "{text}: {:?}", h.painted_text());
    }
    for w in [
        "button:verCopy:contentHash",
        "button:verCopy:pixelHash",
        "button:verCopy:phash",
        "button:verCopy:space",
        "verRecord:json",
        "button:verCopyJson",
    ] {
        assert!(has_widget(&h, w), "{w}");
    }
    assert!(!has_widget(&h, "button:verCopy:pdq"), "this Record has no PDQ: nothing to copy");
    shot(&mut h, "ver-show-record.png");
    let out = dir.path("saved.ver.json");
    dialogs.answer(&out);
    click(&mut h, "button:verSave");
    let req = dialogs.last();
    assert_eq!(req.kind, PickKind::Save);
    assert_eq!(req.file_name.as_deref(), Some("6f9902b45f5f277b6bb3a19604e008c7964234259b4f747611e0dca5c2d27cb7.ver.json"));
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(saved, record);
    assert!(toast(&h).starts_with("Saved the VER Record as"), "{}", toast(&h));
}

/// File ▸ Read VER Record…: a Record of the standard's conformance suite is validated (accepted, its issues listed)
/// and matched against the library (nothing here matches it); one that breaks the rules is rejected with its issue
/// codes. Photos still being fingerprinted are counted.
#[test]
fn read_ver_record_validates_and_reports_the_matches() {
    let (mut h, dialogs) = app(&[]);
    let valid = conformance_record("valid/l3-full-1-1.json").to_string_lossy().to_string();
    dialogs.answer(&valid);
    ask(&mut h, "ui.menu.invoke", json!({"id": "file.readVerRecord"}));
    assert_eq!(dialogs.last().filter.as_ref().map(|f| f.1), Some(&["json"][..]));
    let Some(Dialog::VerImport { file, result, error }) = h.app.ui.dialog.clone() else { panic!("no Read VER Record dialog: {:?}", h.app.ui.dialog) };
    assert_eq!((file.as_str(), error), ("l3-full-1-1.json", None));
    assert_eq!(result["status"], "accepted", "{result}");
    assert!(painted(&h, "Record accepted"), "{:?}", h.painted_text());
    assert!(painted(&h, "It passes its schema and the VER checks."));
    let issues = result["issues"].as_array().unwrap();
    match issues.first() {
        Some(first) => assert!(painted(&h, first["code"].as_str().unwrap()), "the issues are listed"),
        None => assert!(has_widget(&h, "verImport:noIssues")),
    }
    assert_eq!(result["matches"], json!([]));
    assert!(painted(&h, "No photo here matches it."));
    assert!(painted(&h, "9f8fb2dd72cfdce8cae02bec4ca3883bb61a59672d478cafa56687c909d899b9 · 128 × 48"), "the summary: {:?}", h.painted_text());
    shot(&mut h, "ver-read-record.png");

    // while photos are still being fingerprinted, the dialog says matches may be incomplete
    let mut pending = result.clone();
    pending["pendingDigests"] = json!(3);
    h.app.ui.dialog = Some(Dialog::VerImport { file: file.clone(), result: pending, error: None });
    h.settle(SETTLE);
    assert!(painted(&h, "3 photos are still being fingerprinted; matches may be incomplete"));

    // matches are grouped by how they match, and Select Matches selects them
    let ids: Vec<u64> = h.app.session.catalog.photos().take(3).map(|p| p.id.0).collect();
    let mut matched = result.clone();
    matched["matches"] = json!([{"id": ids[0], "by": "content", "score": 1}, {"id": ids[1], "by": "pixel_hash", "score": 1}, {"id": ids[2], "by": "pdq", "score": 12}]);
    h.app.ui.dialog = Some(Dialog::VerImport { file, result: matched, error: None });
    h.settle(SETTLE);
    for text in ["Same file", "Same pixels (a grouping, not identity)", "Looks alike", "PDQ: 12 of 256 bits apart"] {
        assert!(painted(&h, text), "{text}");
    }
    click(&mut h, "button:verSelectMatches");
    let mut selected: Vec<u64> = h.app.session.selection.ids.iter().map(|id| id.0).collect();
    selected.sort();
    assert_eq!(selected, ids);

    // a Record that breaks the rules (one of the suite's invalid Records the engine rejects): rejected, with its issues
    let mut invalid: Vec<PathBuf> = std::fs::read_dir(conformance_record("invalid")).unwrap().filter_map(Result::ok).map(|e| e.path()).collect();
    invalid.sort();
    let bad = invalid
        .into_iter()
        .find(|p| h.app.session.execute("ver.import", &json!({"path": p.to_string_lossy()})).is_ok_and(|r| r["status"] == "rejected"))
        .expect("an invalid Record the engine rejects");
    ask(&mut h, "ui.menu.invoke", json!({"id": "file.readVerRecord", "params": {"path": bad.to_string_lossy()}}));
    let Some(Dialog::VerImport { result, .. }) = h.app.ui.dialog.clone() else { panic!("no dialog") };
    assert_eq!(result["status"], "rejected", "{result}");
    assert!(painted(&h, "Record rejected"));
    let code = result["issues"]
        .as_array()
        .and_then(|i| i.iter().find(|i| i["severity"].as_str().is_some_and(|s| s.eq_ignore_ascii_case("error"))))
        .and_then(|i| i["code"].as_str())
        .unwrap();
    assert!(painted(&h, code), "{code}");
    shot(&mut h, "ver-read-record-rejected.png");
    // and a file that is no Record says so in the dialog
    let not_json = std::env::temp_dir().join(format!("lightcraft-ver-ui-not-json-{}.json", std::process::id()));
    std::fs::write(&not_json, "not json").unwrap();
    let r = h.request("ui.menu.invoke", json!({"id": "file.readVerRecord", "params": {"path": not_json.to_string_lossy()}}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false);
    assert!(has_widget(&h, "verImport:error"));
    let _ = std::fs::remove_file(not_json);
}

/// File ▸ Write VER Records… for the selection: the folder comes from a folder dialog, the work happens on the engine's
/// VER thread under its "Writing VER records" row, and when it is done a toast gives the counts with "Show skipped",
/// which lists each photo and why it has no Record (a demo scene: no file; the JPEG: no model).
#[test]
fn write_ver_records_reports_the_skipped_with_their_reasons() {
    let dir = Scratch::new("write");
    let jpeg = dir.path("pier.jpg");
    write_jpeg(&jpeg, 640, 480);
    let out = dir.path("records");
    let (mut h, dialogs) = app(std::slice::from_ref(&jpeg));
    let pier = photo_named(&h, "pier.jpg");
    let demo: Vec<u64> =
        h.app.session.catalog.photos().filter(|p| matches!(p.source, lightcraft_catalog::Source::Demo { .. })).take(2).map(|p| p.id.0).collect();
    run(&mut h, "library.select", json!({"ids": [demo[0], demo[1], pier.0]}));
    dialogs.answer(&out);
    let file_menu = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "File").expect("File menu").1;
    assert!(file_menu.iter().any(|n| matches!(n, crate::menubar::MenuNode::Item { id, .. } if id == "library.writeVerRecords")));
    h.app.run("library.writeVerRecords", json!({})).unwrap();
    assert_eq!(dialogs.last().kind, PickKind::Folder);
    // the folder dialog answers on the next frame; the JPEG is worked on in the background
    let done = h.step_until(SETTLE, |h| h.app.pending_picks.is_empty() && h.app.caches.ver.batch.is_none() && h.app.caches.ver.skipped.len() == 3);
    assert!(done, "the batch never finished: {:?}", h.app.caches.ver.skipped);
    h.settle(SETTLE);
    let text = toast(&h);
    assert!(text.starts_with("Wrote VER Records for 0 of 3 photos to"), "{text}");
    assert!(text.ends_with(" · 3 skipped"), "{text}");
    assert!(h.app.ui.toast_action.as_ref().is_some_and(|a| a.command == "dialog.verSkipped"));
    click(&mut h, "button:toastAction");
    let Some(Dialog::VerSkipped { rows }) = h.app.ui.dialog.clone() else { panic!("no skipped list: {:?}", h.app.ui.dialog) };
    assert_eq!(rows.len(), 3);
    let pier_row = rows.iter().find(|r| r.0 == "pier.jpg").expect("the JPEG");
    assert!(pier_row.1.contains("search-by-meaning model"), "{pier_row:?}");
    assert_eq!(rows.iter().filter(|r| r.1.starts_with("A generated demo photo")).count(), 2);
    assert!(painted(&h, "pier.jpg") && painted(&h, "A generated demo photo"));
    shot(&mut h, "ver-write-skipped.png");
}

/// The engine's "Writing VER records" row is up while a Write VER Records works.
#[test]
fn write_ver_records_shows_the_engines_activity_row() {
    let dir = Scratch::new("row");
    let jpeg = dir.path("quay.jpg");
    write_jpeg(&jpeg, 640, 480);
    let (mut h, _) = app(std::slice::from_ref(&jpeg));
    let quay = photo_named(&h, "quay.jpg");
    let ctx = h.view.ctx.clone();
    crate::panels::ver::write_records(&mut h.app, &ctx, &dir.path("records"), Some(vec![quay.0])).unwrap();
    assert!(h.app.caches.ver.batch.is_some());
    let mut seen = false;
    let deadline = Instant::now() + SETTLE;
    while h.app.caches.ver.batch.is_some() {
        h.step();
        seen |= h.app.session.activity.list().iter().any(|t| t.label == "Writing VER records");
        assert!(Instant::now() < deadline);
    }
    assert!(seen, "the engine's row showed the work");
    assert_eq!(h.app.caches.ver.skipped.len(), 1);
    assert!(h.app.session.activity.list().iter().all(|t| t.label != "Writing VER records"), "the row went with the work");
}

/// The export dialog's "Write VER Records" starts from Settings, applies to the export it is set for, leaves Settings
/// as they were, and the export's toast says once which files got no Record and why.
#[test]
fn the_export_dialog_writes_ver_records_as_chosen_and_says_which_files_got_none() {
    let dir = Scratch::new("export");
    let (mut h, _) = app(&[]);
    let export = |h: &mut Headless, on: Option<bool>, into: &str| {
        ask(h, "ui.menu.invoke", json!({"id": "dialog.export"}));
        assert!(has_widget(h, "check:exportVerRecords"), "the dialog has the checkbox");
        if let Some(Dialog::Export { ver_records, dir, full_size, .. }) = &mut h.app.ui.dialog {
            assert_eq!(*ver_records, Some(true), "it starts from Settings");
            if let Some(on) = on {
                *ver_records = Some(on);
            }
            *dir = into.to_string();
            *full_size = false;
        }
        h.settle(SETTLE);
        click(h, "button:dialogOk");
        let finished = h.step_until(SETTLE, |h| h.app.export.is_none() && h.app.last_export_result.is_some());
        assert!(finished, "the export never finished");
        h.settle(SETTLE);
        let files = h.app.last_export_result.take().unwrap()["files"].clone();
        (files, toast(h))
    };
    let (files, text) = export(&mut h, None, &dir.path("on"));
    assert!(files[0]["path"].is_string(), "{files}");
    assert!(text.contains("VER: 1 of 1 files got no Record: needs the model"), "{text}");
    let (files, text) = export(&mut h, Some(false), &dir.path("off"));
    assert!(files[0]["notes"].is_null(), "no Records asked for: {files}");
    assert!(!text.contains("VER:"), "{text}");
    assert_eq!(run(&mut h, "ver.config", json!({}))["exportSidecars"], true, "Settings are as they were");
}

/// Show Duplicates names each duplicate's tier on its cell: two files of the same pixels (one with a print resolution,
/// one without) are "Same pixels", between "Same file" and the fingerprint tiers.
#[test]
fn the_duplicates_view_paints_same_pixels() {
    let dir = Scratch::new("dups");
    let (a, b) = (dir.path("lake.png"), dir.path("lake (300 ppi).png"));
    write_png(&a, None);
    write_png(&b, Some(300));
    assert_ne!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), "two files");
    let (mut h, _) = app(&[a, b]);
    let (a, b) = (photo_named(&h, "lake.png"), photo_named(&h, "lake (300 ppi).png"));
    let status = run(&mut h, "vera.status", json!({}));
    assert!(status["pixelHashed"].as_u64().unwrap() >= 2, "{status}");
    run(&mut h, "library.select", json!({"ids": [a.0]}));
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showDuplicates"}));
    assert_eq!(r["photos"][0]["id"], b.0, "{r}");
    assert_eq!(r["photos"][0]["tier"], "pixels", "{r}");
    assert!(has_widget(&h, &format!("badge:duplicate:{}", b.0)), "the duplicate's cell names its tier");
    assert!(!has_widget(&h, &format!("badge:duplicate:{}", a.0)), "the photo itself has none");
    assert!(painted(&h, "Same pixels"), "{:?}", h.painted_text());
    shot(&mut h, "ver-duplicates-same-pixels.png");
    assert_eq!(crate::panels::vera::tier_label("bytes"), Some("Same file"));
    assert_eq!(crate::panels::vera::tier_label("pixels"), Some("Same pixels"));
    // Settings ▸ Search counts them
    run(&mut h, "app.settings", json!({"tab": "search"}));
    let counts = h.painted_text().into_iter().find(|t| t.contains("photos fingerprinted")).unwrap_or_default();
    assert!(counts.contains(&format!("pixel-hashed {}", status["pixelHashed"])), "{counts}");
    // (in a window of ordinary height the tab scrolls)
    shot(&mut h, "ver-settings-scrolling.png");
}
