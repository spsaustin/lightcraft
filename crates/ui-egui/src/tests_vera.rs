//! Headless tests of VERA in the app (`panels::vera`, `docs/vera.md`): background fingerprinting at the face scan's
//! pace, the ranked search the search field runs on Enter (and its chip), and Photo ▸ Show Duplicates.

use std::time::Duration;

use lightcraft_catalog::{Op, Photo, SortKey, Source};
use serde_json::{Value, json};

use crate::headless::Headless;
use crate::state::ViewMode;
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

fn ask(h: &mut Headless, method: &str, params: Value) -> Value {
    let r = h.request(method, params.clone(), T);
    assert_eq!(r["ok"], true, "{method} {params}: {r}");
    h.settle(SETTLE);
    r["result"].clone()
}

fn run(h: &mut Headless, command: &str, params: Value) -> Value {
    ask(h, "engine.execute", json!({"command": command, "params": params}))
}

/// The chip labels the strip above the grid shows, as the app words them.
fn chip_labels(h: &Headless) -> Vec<String> {
    let chips = lightcraft_engine::filter_chips(&h.app.session.filter, &h.app.session.catalog);
    chips.iter().map(|c| crate::panels::chips::label(&h.app, c)).collect()
}

/// The index of the chip that clears the filter's `only` list (its × is `chip:<i>`).
fn only_chip(h: &Headless) -> usize {
    let chips = lightcraft_engine::filter_chips(&h.app.session.filter, &h.app.session.catalog);
    chips.iter().position(|c| c.clear.get("only").is_some()).expect("an only chip")
}

fn painted(h: &Headless, text: &str) -> bool {
    h.painted_text().iter().any(|t| t.contains(text))
}

/// Every photo of the demo library gets its fingerprint in the background, within a bounded number of frames, and
/// `ui.inspect` → `vera` says so (next to `faceScan`).
#[test]
fn vera_fingerprints_the_demo_library_in_the_background() {
    let mut h = demo();
    assert!(h.app.session.catalog.len() > 1);
    let done = |h: &Headless| h.app.caches.vera_pending == 0 && h.app.caches.vera_digested > 0;
    let mut frames = 0;
    while !done(&h) && frames < 3000 {
        h.step();
        frames += 1;
    }
    assert!(done(&h), "after {frames} frames: {} pending, {} digested", h.app.caches.vera_pending, h.app.caches.vera_digested);
    let v = ask(&mut h, "ui.inspect", json!({}))["vera"].clone();
    assert_eq!(v["pending"], 0, "{v}");
    assert!(["pause", "light", "normal", "full"].contains(&v["pace"].as_str().unwrap_or("")), "{v}");
    let status = run(&mut h, "vera.status", json!({}));
    assert_eq!(v["digested"], status["photos"], "every photo has its fingerprint: {v} {status}");
    // the engine agrees, and the activity row went with the work
    assert_eq!(status["pendingDigests"], 0, "{status}");
    assert!(h.app.session.activity.list().iter().all(|t| t.kind != "vera"), "{:?}", h.app.session.activity.list());
}

/// Enter in the search field ranks the photos by its words (`search.text`): the photo titled with them comes first
/// under a "Ranked" chip, with the relevance sort; the chip's × gives back the order from before, and so does Clear
/// Filters. Typing still filters live, and a search that finds nothing says so and leaves the grid alone.
#[test]
fn enter_in_the_search_field_ranks_and_the_chip_puts_the_order_back() {
    let mut h = fingerprinted();
    let before = h.app.session.visible_cloned();
    let sort = h.app.session.sort;
    assert_ne!(sort.key, SortKey::Relevance);
    // the last photo in the grid, given a title nothing else has
    let target = *before.last().expect("demo photos");
    run(&mut h, "photo.setMeta", json!({"ids": [target.0], "title": "Zanzibar lighthouse at dusk"}));
    let before = h.app.session.visible_cloned();
    ask(&mut h, "ui.clickWidget", json!({"id": "field:search"}));
    ask(&mut h, "ui.text", json!({"text": "zanzibar"}));
    assert_eq!(h.app.session.filter.text, "zanzibar", "typing filters live as before");
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    let ranked = h.app.session.visible_cloned();
    assert_eq!(ranked.first(), Some(&target), "the titled photo first: {ranked:?}");
    assert_eq!(h.app.session.active(), Some(target), "the best match is the active photo (the one shown was left out)");
    assert_eq!(h.app.session.sort.key, SortKey::Relevance);
    assert_eq!(h.app.session.filter.text, "", "the ranking replaces the live filter");
    assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "Enter in the field opens nothing else");
    assert_eq!(chip_labels(&h), ["Ranked: “zanzibar”"]);
    assert!(painted(&h, "Ranked: “zanzibar”"), "the chip is drawn");
    // × on the chip: everything back as it was
    let chip = only_chip(&h);
    ask(&mut h, "ui.clickWidget", json!({"id": format!("chip:{chip}")}));
    assert!(h.app.session.filter.only.is_empty());
    assert_eq!(h.app.session.sort, sort, "the sort from before the search");
    assert_eq!(h.app.session.visible_cloned(), before, "the order from before the search");
    assert!(h.app.ui.only_origin.is_none() && h.app.ui.search.is_empty());
    assert!(chip_labels(&h).is_empty());

    // again, and View ▸ Clear Filters does the same
    ask(&mut h, "ui.clickWidget", json!({"id": "field:search"}));
    ask(&mut h, "ui.text", json!({"text": "zanzibar"}));
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    assert_eq!(h.app.session.sort.key, SortKey::Relevance);
    run(&mut h, "library.clearFilter", json!({}));
    assert_eq!(h.app.session.sort, sort);
    assert_eq!(h.app.session.visible_cloned(), before);

    // typing after a ranked search filters live again (the ranking goes)
    ask(&mut h, "ui.clickWidget", json!({"id": "field:search"}));
    ask(&mut h, "ui.text", json!({"text": "zanzibar"}));
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    ask(&mut h, "ui.clickWidget", json!({"id": "field:search"}));
    ask(&mut h, "ui.key", json!({"key": "Backspace"}));
    assert!(h.app.session.filter.only.is_empty() && h.app.ui.only_origin.is_none());
    assert_eq!((h.app.session.filter.text.as_str(), h.app.session.sort), ("zanziba", sort));
    run(&mut h, "library.clearFilter", json!({}));
    h.app.ui.search.clear();

    // nothing found: said so, and the grid stays as it was
    ask(&mut h, "ui.clickWidget", json!({"id": "field:search"}));
    ask(&mut h, "ui.text", json!({"text": "qqxyzzy"}));
    run(&mut h, "library.filter", json!({"text": ""}));
    ask(&mut h, "ui.key", json!({"key": "Enter"}));
    let toast = h.app.ui.toast.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    assert_eq!(toast, "No photos match “qqxyzzy”");
    assert!(h.app.session.filter.only.is_empty() && h.app.ui.only_origin.is_none());
    assert_eq!(h.app.session.sort, sort);
}

/// Photo ▸ Show Duplicates is a Photo menu item, enabled with an active photo. It shows the photo and its duplicates
/// under a "Duplicates of …" chip whose × shows everything again; a photo with none says so and changes nothing.
#[test]
fn show_duplicates_filters_to_the_photo_and_its_copies() {
    let mut h = fingerprinted();
    let photo_menu = crate::menubar::menu_bar(&h.app).into_iter().find(|(t, _)| t == "Photo").expect("Photo menu").1;
    let item = photo_menu.iter().find_map(|n| match n {
        crate::menubar::MenuNode::Item { id, label, enabled, .. } if id == "photo.showDuplicates" => Some((label.clone(), *enabled)),
        _ => None,
    });
    assert_eq!(item, Some(("Show Duplicates".to_string(), true)), "in the Photo menu, enabled with an active photo");
    let active = h.app.session.selection.clone();
    h.app.session.selection = Default::default();
    assert!(!crate::menus::ui_enabled(&h.app, "photo.showDuplicates"), "no active photo: disabled");
    h.app.session.selection = active;

    // the demo scenes are all different pictures: none found, nothing changes
    let id = h.app.session.active().expect("an active photo");
    let name = h.app.session.catalog.photo(id).map(|p| p.file_name.clone()).unwrap_or_default();
    let before = h.app.session.visible_cloned();
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showDuplicates"}));
    assert_eq!(r["photos"], json!([]), "{r}");
    assert!(h.app.session.filter.only.is_empty() && h.app.ui.only_origin.is_none());
    assert_eq!(h.app.session.visible_cloned(), before);
    let toast = h.app.ui.toast.as_ref().map(|t| t.0.clone()).unwrap_or_default();
    assert!(toast.starts_with(&format!("No duplicates of {name} found")), "{toast}");

    // two files with the same bytes (the same content hash): duplicates by the engine's first tier
    let mut add = |n: u64, file: &str| {
        let pid = h.app.session.catalog.alloc_photo_id();
        let mut p = Photo::new(pid, Source::File { path: format!("/lightcraft-vera-test/{file}") }, file, "JPEG", 600, 400, "2026-01-01T00:00:00");
        p.content_hash = Some("same-bytes".into());
        p.captured = Some(format!("2020-01-0{n}T00:00:00"));
        h.app.session.catalog.apply(Op::AddPhoto { photo: Box::new(p) }).unwrap();
        pid
    };
    let (a, b) = (add(1, "original.jpg"), add(2, "copy of original.jpg"));
    run(&mut h, "library.select", json!({"ids": [a.0]}));
    let r = ask(&mut h, "ui.menu.invoke", json!({"id": "photo.showDuplicates"}));
    assert_eq!(r["photos"][0]["id"], b.0, "{r}");
    assert_eq!(r["photos"][0]["tier"], "bytes", "{r}");
    assert_eq!(h.app.session.filter.only, [a, b]);
    let mut shown = h.app.session.visible_cloned();
    shown.sort();
    assert_eq!(shown, [a, b]);
    assert_eq!(chip_labels(&h), ["Duplicates of original.jpg"]);
    assert!(painted(&h, "Duplicates of original.jpg"), "the chip is drawn");
    let chip = only_chip(&h);
    ask(&mut h, "ui.clickWidget", json!({"id": format!("chip:{chip}")}));
    assert!(h.app.session.filter.only.is_empty() && h.app.ui.only_origin.is_none());
    assert_eq!(h.app.session.visible_cloned().len(), before.len() + 2);

    // the grid's context menu offers it too
    let canvas = h.app.canvas_rect.expect("the grid");
    let r = h.app.widgets.iter().find(|(w, r)| w.starts_with("thumb:") && canvas.contains_rect(*r)).map(|(_, r)| *r).expect("a photo's cell");
    ask(&mut h, "ui.click", json!({"x": r.center().x, "y": r.center().y, "button": "right"}));
    assert!(egui::Popup::is_any_open(&h.view.ctx) && painted(&h, "Show Duplicates"), "in the context menu");
}
