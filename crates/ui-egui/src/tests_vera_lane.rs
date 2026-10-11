//! Headless tests of VERA's vector lane in the app (`panels::vera_settings`, `panels::vera`, `docs/vera.md`):
//! Settings ▸ Search with its lanes and the model download behind its licence dialog, Find Photos That Look Alike, and
//! what the ranked chip and the duplicates toast say about search by meaning. No test downloads anything; the one that
//! uses the real model runs only with `LIGHTCRAFT_SIGLIP2_DIR` set to its checkpoint.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::headless::Headless;
use crate::state::{Dialog, OnlyKind, ViewMode};
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(30);
const SETTLE: Duration = Duration::from_secs(120);

fn demo() -> Headless {
    let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), Services { png: None, ..Default::default() });
    app.ui.view = ViewMode::PhotoGrid;
    let mut h = Headless::new(app, [1400.0, 900.0], 1.0);
    h.settle(SETTLE);
    h
}

/// The demo app once background fingerprinting is done (its activity row sits over the top of the grid until then).
fn fingerprinted() -> Headless {
    let mut h = demo();
    let done = h.step_until(SETTLE, |h| h.app.caches.vera_pending == 0 && h.app.caches.vera_digested > 0);
    assert!(done, "fingerprinting never finished: {} pending", h.app.caches.vera_pending);
    h.settle(SETTLE);
    h
}

/// An empty folder for the model (as the desktop app has one): nothing is ever downloaded into it.
fn model_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lightcraft-vera-ui-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("siglip2-base-patch32-256")
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
    ask(h, "ui.clickWidget", json!({"id": id}));
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

fn chip_labels(h: &Headless) -> Vec<String> {
    let chips = lightcraft_engine::filter_chips(&h.app.session.filter, &h.app.session.catalog);
    chips.iter().map(|c| crate::panels::chips::label(&h.app, c)).collect()
}

/// The Photo menu's entry for `id`: (label, enabled).
fn photo_menu_item(h: &Headless, id: &str) -> Option<(String, bool)> {
    let photo = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "Photo").expect("Photo menu").1;
    photo.iter().find_map(|n| match n {
        crate::menubar::MenuNode::Item { id: i, label, enabled, .. } if i == id => Some((label.clone(), *enabled)),
        _ => None,
    })
}

/// Settings ▸ Search lists the four lanes in the states `vera.status` gives them (on a demo session: words and
/// fingerprints on, meaning waiting for its model, describe not yet), the counts, and the model's download button —
/// enabled when there is a folder to keep the model in, disabled (and saying why) when there is none.
#[test]
fn vera_settings_search_tab_shows_the_lanes_and_the_download() {
    let mut h = demo();
    let dir = model_dir("tab");
    h.app.session.vera_model_dir = Some(dir.clone());
    let status = run(&mut h, "vera.status", json!({}));
    assert_eq!(status["lanes"]["keyword"], "live", "{status}");
    assert_eq!(status["lanes"]["digest"], "live", "{status}");
    assert_eq!(status["lanes"]["vector"]["state"], "needs_model", "{status}");
    let describe = status["lanes"]["describe"]["state"].as_str().unwrap_or("").to_string();
    assert!(!describe.is_empty(), "{status}");

    run(&mut h, "app.settings", json!({"tab": "search"}));
    assert_eq!(h.app.ui.dialog, Some(Dialog::Settings { tab: "search".into() }));
    assert!(has_widget(&h, "button:settingsTab-search"), "a tab of its own");
    for lane in ["keyword", "digest", "vector", "describe"] {
        assert!(has_widget(&h, &format!("vera:lane:{lane}")), "the {lane} lane has a row");
    }
    for text in
        ["By words", "Same picture", "By meaning", "Describe", "Needs its model", "one-time download (below)", "so search by words finds them"]
    {
        assert!(painted(&h, text), "{text:?} is drawn: {:?}", h.painted_text());
    }
    // the two live lanes say so, and the describe lane says what it waits for
    let on = if describe == "live" { 3 } else { 2 };
    assert_eq!(h.painted_text().iter().filter(|t| t.as_str() == "On").count(), on, "{:?}", h.painted_text());
    assert!(painted(&h, crate::panels::vera_settings::state_label(&describe)), "{describe}: {:?}", h.painted_text());
    let photos = status["photos"].as_u64().unwrap();
    assert!(painted(&h, &format!("of {photos} photos fingerprinted")), "{:?}", h.painted_text());
    assert!(painted(&h, "Download the model (1.5 GB)"));
    assert!(has_widget(&h, "button:veraDownload"));
    assert!(!painted(&h, "nowhere to keep the model"));

    // no folder for the model (the CLI, MCP): the button is there but cannot be used, and the tab says why
    h.app.session.vera_model_dir = None;
    h.app.caches.vera_epoch += 1;
    h.settle(SETTLE);
    assert!(painted(&h, "nowhere to keep the model"), "{:?}", h.painted_text());
    click(&mut h, "button:veraDownload");
    assert_eq!(h.app.ui.dialog, Some(Dialog::Settings { tab: "search".into() }), "a disabled button opens nothing");
    assert!(!dir.exists(), "nothing was written");
}

/// Download opens the model's terms (name, size, the Apache-2.0 notice, a link to the licence, the accept box). OK
/// waits for the box; the engine refuses a download whose licence was not accepted too. Accepted, the download is
/// asked for — here in a session with nowhere to keep the model, so nothing is fetched and the dialog shows why.
#[test]
fn vera_licence_dialog_refuses_until_the_terms_are_accepted() {
    let mut h = demo();
    h.app.session.vera_model_dir = Some(model_dir("licence"));
    run(&mut h, "app.settings", json!({"tab": "search"}));
    click(&mut h, "button:veraDownload");
    let Some(Dialog::VeraModel { info, accepted: false, error: None }) = h.app.ui.dialog.clone() else {
        panic!("the licence dialog: {:?}", h.app.ui.dialog)
    };
    assert_eq!(info["licence"]["name"], "Apache License 2.0", "{info}");
    for text in [
        "Download the SigLIP 2 Model",
        "SigLIP 2 Base",
        "1.5 GB",
        "Apache License 2.0 · commercial use allowed",
        "Copyright 2025 Google LLC",
        "I have read these terms and accept them for my own use",
        "Nothing is fetched until you accept",
        "kept on this computer only",
        "never uploads",
    ] {
        assert!(painted(&h, text), "{text:?} is drawn: {:?}", h.painted_text());
    }
    assert!(has_widget(&h, "link:veraLicence") && has_widget(&h, "link:veraApache"));
    // OK waits for the box
    click(&mut h, "button:dialogOk");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::VeraModel { accepted: false, .. })), "{:?}", h.app.ui.dialog);
    let r = h.request("ui.dialog.confirm", json!({}), T);
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["error"].as_str().unwrap_or("").contains("Tick the box"), "{r}");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::VeraModel { .. })), "stays open");
    // and the engine refuses without the acceptance, naming the licence
    for p in [json!({}), json!({"acceptLicense": false})] {
        let r = h.request("engine.execute", json!({"command": "vera.model.download", "params": p}), T);
        let e = r["error"].as_str().unwrap_or("");
        assert!(r["ok"] == false && e.contains("Apache License 2.0") && e.contains("acceptLicense"), "{r}");
    }
    assert!(run(&mut h, "vera.model.status", json!({}))["downloading"].is_null(), "nothing started");

    // ticked: OK asks for the download (in a session with nowhere to keep it, so no network is touched)
    click(&mut h, "check:veraModel.accept");
    assert!(matches!(h.app.ui.dialog, Some(Dialog::VeraModel { accepted: true, .. })));
    h.app.session.vera_model_dir = None;
    click(&mut h, "button:dialogOk");
    let Some(Dialog::VeraModel { error: Some(e), .. }) = h.app.ui.dialog.clone() else { panic!("stays open with the error: {:?}", h.app.ui.dialog) };
    assert!(e.contains("nowhere to keep it"), "{e}");
    assert!(painted(&h, "The download didn't work:"), "{:?}", h.painted_text());
    assert!(!h.app.caches.vera_dl_watch);
    click(&mut h, "button:dialogCancel");
    assert_eq!(h.app.ui.dialog, None);
}

/// Photo ▸ Find Photos That Look Alike is a Photo menu item (and on the grid's context menu), enabled with an active
/// photo. Without the model it changes nothing and says why in a toast whose button opens Settings ▸ Search; with
/// nowhere to keep a model the toast says that instead, with no button. Show Duplicates' "none found" mentions that
/// search by meaning is off.
#[test]
fn vera_look_alikes_without_the_model_says_what_it_needs() {
    let mut h = fingerprinted();
    assert_eq!(photo_menu_item(&h, "photo.showLookAlikes"), Some(("Find Photos That Look Alike".to_string(), true)));
    assert!(crate::menus::UI_COMMANDS.iter().any(|c| c.0 == "photo.showLookAlikes"));
    let active = h.app.session.selection.clone();
    h.app.session.selection = Default::default();
    assert!(!crate::menus::ui_enabled(&h.app, "photo.showLookAlikes"), "no active photo: disabled");
    h.app.session.selection = active;

    h.app.session.vera_model_dir = Some(model_dir("alike"));
    let before = h.app.session.visible_cloned();
    let r = h.request("ui.menu.invoke", json!({"id": "photo.showLookAlikes"}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false, "{r}");
    assert!(r["error"].as_str().unwrap_or("").contains("the vector lane is not ready"), "agents get the engine's error: {r}");
    assert_eq!(toast(&h), "Finding photos that look alike needs the SigLIP 2 model: a one-time download, in Settings → Search.");
    assert!(h.app.session.filter.only.is_empty() && h.app.ui.only_origin.is_none());
    assert_eq!(h.app.session.visible_cloned(), before, "the view is left alone");
    // the toast's button opens Settings ▸ Search
    assert!(has_widget(&h, "button:toastAction") && painted(&h, "Open Settings → Search"), "{:?}", h.painted_text());
    click(&mut h, "button:toastAction");
    assert_eq!(h.app.ui.dialog, Some(Dialog::Settings { tab: "search".into() }));
    assert!(h.app.ui.toast.is_none() && h.app.ui.toast_action.is_none());
    h.app.ui.dialog = None;

    // Show Duplicates finding none says that crops and look-alikes need search by meaning
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showDuplicates"}));
    assert_eq!(r["photos"], json!([]), "{r}");
    assert!(toast(&h).ends_with(" · search by meaning is off: Settings → Search"), "{}", toast(&h));
    assert!(h.app.ui.toast_action.is_some());

    // nowhere to keep a model: the engine's reason, and nothing to open
    h.app.session.vera_model_dir = None;
    let r = h.request("ui.menu.invoke", json!({"id": "photo.showLookAlikes"}), T);
    h.settle(SETTLE);
    assert_eq!(r["ok"], false);
    assert!(toast(&h).contains("nowhere to keep it"), "{}", toast(&h));
    assert!(h.app.ui.toast_action.is_none());
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showDuplicates"}));
    assert_eq!(r["photos"], json!([]));
    assert!(!toast(&h).contains("by meaning"), "nothing the user could switch on: {}", toast(&h));

    // the grid's context menu offers it next to Find Similar Photos
    let canvas = h.app.canvas_rect.expect("the grid");
    let r = h.app.widgets.iter().find(|(w, r)| w.starts_with("thumb:") && canvas.contains_rect(*r)).map(|(_, r)| *r).expect("a photo's cell");
    ask(&mut h, "ui.click", json!({"x": r.center().x, "y": r.center().y, "button": "right"}));
    assert!(egui::Popup::is_any_open(&h.view.ctx) && painted(&h, "Find Photos That Look Alike"), "in the context menu");
    assert!(painted(&h, "Find Similar Photos"));
}

/// The ranked chip says when the vector lane ranked the results too; without it, it reads as in Phase 1.
#[test]
fn vera_ranked_chip_names_the_lanes() {
    let mut h = fingerprinted();
    let target = *h.app.session.visible_cloned().last().expect("demo photos");
    run(&mut h, "photo.setMeta", json!({"ids": [target.0], "title": "Zanzibar lighthouse at dusk"}));
    click(&mut h, "field:search");
    ask(&mut h, "ui.text", json!({"text": "zanzibar"}));
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    assert_eq!(chip_labels(&h), ["Ranked: “zanzibar”"], "words alone");
    assert!(matches!(&h.app.ui.only_origin, Some(o) if o.kind == OnlyKind::Ranked { query: "zanzibar".into(), meaning: false }));
    // the same results ranked by both lanes (what search.text reports as lanes: ["keyword", "vector"])
    if let Some(o) = h.app.ui.only_origin.as_mut() {
        o.kind = OnlyKind::Ranked { query: "zanzibar".into(), meaning: true };
    }
    h.settle(SETTLE);
    assert_eq!(chip_labels(&h), ["Ranked: “zanzibar” · by words and meaning"]);
    assert!(painted(&h, "Ranked: “zanzibar” · by words and meaning"), "the chip is drawn");
}

/// With the real model (`LIGHTCRAFT_SIGLIP2_DIR`): the library is embedded in the background, Find Photos That Look
/// Alike on the lake scene shows it and the other lake scene under a "Looks like …" chip whose × gives the view back,
/// and a ranked search runs on both lanes. Skipped without the checkpoint.
#[test]
fn vera_look_alikes_with_the_real_model() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR") else {
        eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
        return;
    };
    let mut h = fingerprinted();
    h.app.session.vera_model_dir = Some(PathBuf::from(dir));
    let t0 = Instant::now();
    let status = loop {
        for _ in 0..20 {
            h.step();
            std::thread::sleep(Duration::from_millis(2));
        }
        let st = h.app.session.execute("vera.status", &json!({})).unwrap();
        if st["embedded"] == st["photos"] {
            break st;
        }
        assert!(t0.elapsed() < Duration::from_secs(900), "embedding never finished: {st}");
    };
    eprintln!("embedded {} photos in {:.1} s", status["embedded"], t0.elapsed().as_secs_f64());
    assert_eq!(status["lanes"]["vector"]["state"], "live", "{status}");
    h.settle(SETTLE);

    // Settings ▸ Search: the switch, the counts and Remove (not pressed)
    run(&mut h, "app.settings", json!({"tab": "search"}));
    assert!(has_widget(&h, "check:vera.vectorLane") && has_widget(&h, "button:veraRemove"));
    assert!(painted(&h, &format!("{0} of {0} embedded", status["photos"])), "{:?}", h.painted_text());
    h.app.ui.dialog = None;
    h.settle(SETTLE);

    let lake = h
        .app
        .session
        .catalog
        .photos()
        .find(|p| p.meta.title == "Lake at first light")
        .map(|p| (p.id, p.file_name.clone()))
        .expect("the demo's lake scene");
    run(&mut h, "library.select", json!({"ids": [lake.0.0]}));
    let before = h.app.session.visible_cloned();
    let sort = h.app.session.sort;
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showLookAlikes"}));
    assert_eq!(r["filtered"], true, "{r}");
    let shown = h.app.session.visible_cloned();
    assert!(shown.len() >= 2, "the lake and what looks like it: {r}");
    assert_eq!(shown.first(), Some(&lake.0), "the photo first");
    assert_eq!(chip_labels(&h), [format!("Looks like {}", lake.1)]);
    assert!(painted(&h, &format!("Looks like {}", lake.1)), "the chip is drawn");
    // × gives the view and the sort back
    let chips = lightcraft_engine::filter_chips(&h.app.session.filter, &h.app.session.catalog);
    let chip = chips.iter().position(|c| c.clear.get("only").is_some()).expect("an only chip");
    click(&mut h, &format!("chip:{chip}"));
    assert_eq!((h.app.session.visible_cloned(), h.app.session.sort), (before, sort));

    // a ranked search runs on both lanes, and its chip says so
    click(&mut h, "field:search");
    ask(&mut h, "ui.text", json!({"text": "lake"}));
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    assert_eq!(chip_labels(&h), ["Ranked: “lake” · by words and meaning"]);
}

/// Headless screenshots of Settings ▸ Search and the licence dialog, for looking at them: written only when
/// `LIGHTCRAFT_VERA_SHOTS` names a folder. (`lightcraft-cli snapshot` sessions have no folder for the model, so the
/// download button is disabled there and the dialog cannot be opened.)
#[test]
fn vera_screenshots_of_the_search_tab_and_the_licence() {
    let Some(out) = std::env::var_os("LIGHTCRAFT_VERA_SHOTS").map(PathBuf::from) else {
        eprintln!("skipped: set LIGHTCRAFT_VERA_SHOTS to a folder for the screenshots");
        return;
    };
    let services = Services {
        write: Some(Box::new(lightcraft_engine::export::write_file)),
        png: Some(Box::new(|img: &lightcraft_raster::Rgba8| {
            lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(img), &lightcraft_codecs::EncodeMeta::default()).unwrap_or_default()
        })),
        ..Default::default()
    };
    let mut app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), services);
    app.ui.view = ViewMode::PhotoGrid;
    app.session.vera_model_dir = Some(model_dir("shots"));
    let mut h = Headless::new(app, [1600.0, 1000.0], 1.0);
    let done = h.step_until(SETTLE, |h| h.app.caches.vera_pending == 0 && h.app.caches.vera_digested > 0);
    assert!(done);
    h.settle(SETTLE);
    run(&mut h, "app.settings", json!({"tab": "search"}));
    let shot = |h: &mut Headless, name: &str| {
        let path = out.join(name);
        let r = h.request("ui.screenshot", json!({"path": path.to_string_lossy(), "headless": true}), SETTLE);
        assert_eq!(r["ok"], true, "{r}");
        eprintln!("wrote {}", path.display());
    };
    shot(&mut h, "vera-settings-search.png");
    click(&mut h, "button:veraDownload");
    shot(&mut h, "vera-licence-dialog.png");
    click(&mut h, "check:veraModel.accept");
    shot(&mut h, "vera-licence-dialog-accepted.png");
    // Find Photos That Look Alike without the model: the toast and its button
    click(&mut h, "button:dialogCancel");
    let r = h.request("ui.menu.invoke", json!({"id": "photo.showLookAlikes"}), T);
    assert_eq!(r["ok"], false, "{r}");
    shot(&mut h, "vera-look-alikes-toast.png");
}
