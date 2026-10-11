//! VERA in the app (`docs/vera.md`): fingerprinting and embedding the library in the background, the ranked search the
//! search field runs on Enter, Show Duplicates and Find Photos That Look Alike. The engine does the work (`vera.pump`,
//! `search.text`, `photo.duplicates`, `photo.similar`); this module decides when to call it, names what the grid shows
//! in its chip ("Ranked: “dog”", "Duplicates of IMG_0042.JPG", "Looks like IMG_0042.JPG") and puts the view back when
//! those results go. Settings ▸ Search, with the model download, is `panels::vera_settings`.

use lightcraft_catalog::{PhotoId, Sort, SortKey};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::state::{OnlyKind, OnlyOrigin, ViewMode};

/// Called every frame, right after the face scan's pump: fingerprints the next photos at the pace the face scan was
/// given (nothing while the user drags, types or scrolls), and notes how far it has got for `ui.inspect`.
pub fn pump(app: &mut LightcraftApp, ctx: &egui::Context) {
    // results whose `only` list was replaced or cleared some other way (an agent, Find Similar) are over
    sync(app);
    let now = ctx.input(|i| i.time);
    // a model download the user started: followed until it ends
    super::vera_settings::watch(app, ctx, now);
    if now < app.caches.vera_next_pump {
        return;
    }
    let (pace, _) = super::faces::scan_pace(app, ctx, now);
    let Ok(v) = app.session.execute("vera.pump", &json!({"pace": pace})) else { return };
    app.caches.vera_pace = pace;
    app.caches.vera_pending = v["pending"].as_u64().unwrap_or(0);
    app.caches.vera_digested = v["digested"].as_u64().unwrap_or(0);
    // the vector lane, when it runs: its pictures go to the worker and come back through the same call
    app.caches.vera_embedding = (v["embedding"]["pending"].as_u64().unwrap_or(0), v["embedding"]["embedded"].as_u64().unwrap_or(0));
    let photos = v["photos"].as_u64().unwrap_or(0);
    // Describe: photos waiting for a worker or in flight, and the proposals waiting for review
    let d = &v["describe"];
    let describe = (d["pending"].as_u64().unwrap_or(0), d["inFlight"].as_u64().unwrap_or(0), d["proposals"].as_u64().unwrap_or(0));
    // VER Records written in the background (Show VER Record…, Write VER Records…)
    let ver_pending = v["ver"]["pending"].as_u64().unwrap_or(0);
    super::ver::pumped(app, ctx, &v["ver"]);
    let was_busy = app.caches.vera_describe.0 + app.caches.vera_describe.1 > 0;
    app.caches.vera_describe = describe;
    if was_busy && describe.0 + describe.1 == 0 {
        super::describe_review::run_ended(app, ctx);
    }
    // as for the face scan: called again in 50 ms while there is work, looked at again every few seconds once it is
    // done (new photos are fingerprinted soon after they arrive), and nothing asked for while the library is empty
    let work = app
        .caches
        .vera_pending
        .saturating_add(app.caches.vera_embedding.0)
        .saturating_add(describe.0)
        .saturating_add(describe.1)
        .saturating_add(ver_pending);
    let (next, wait) = schedule(work, photos);
    app.caches.vera_next_pump = now + next;
    if let Some(secs) = wait {
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(secs));
    }
}

/// When to call the pump again (seconds from now) and the repaint to ask for, with `pending` photos left of `photos`.
fn schedule(pending: u64, photos: u64) -> (f64, Option<f64>) {
    match (pending > 0, photos > 0) {
        (true, _) => (0.05, Some(0.05)),
        (false, true) => (1.0, Some(5.0)),
        (false, false) => (0.0, None),
    }
}

/// Enter in the search field: rank the photos by these words (`search.text`). The grid shows the results best first
/// (the relevance sort) under a "Ranked" chip; a search that finds nothing leaves the view alone and says so.
pub fn ranked_search(app: &mut LightcraftApp, ctx: &egui::Context, query: &str) {
    let query = query.trim();
    if query.is_empty() {
        return;
    }
    sync(app);
    let sort = return_sort(app);
    let r = app.session.execute("search.text", &json!({"query": query}));
    // the vector lane should have run and could not: said once (the search still ran, by words)
    let warning = r.as_ref().ok().and_then(|r| r["warning"].as_str()).map(str::to_string);
    let warned = match warning {
        Some(w) if app.caches.vera_warning.as_deref() != Some(w.as_str()) => {
            app.toast_error(ctx, w.clone());
            app.caches.vera_warning = Some(w);
            true
        }
        _ => false,
    };
    match r {
        Ok(r) if r["filtered"] == true => {
            let meaning = r["lanes"].as_array().is_some_and(|l| l.iter().any(|x| x == "vector"));
            // the live filter from typing would narrow the ranked list: the ranking replaces it (the field keeps the words)
            if !app.session.filter.text.is_empty() {
                let _ = app.session.execute("library.filter", &json!({"text": ""}));
            }
            let ids = app.session.filter.only.clone();
            // the photo shown was left out: the best match takes its place (in the loupe, and as the grid's active photo)
            if let Some(best) = ids.first().filter(|_| app.session.active().is_none_or(|a| !ids.contains(&a))) {
                let _ = app.session.execute("library.select", &json!({"ids": [best.0]}));
            }
            app.ui.only_origin = Some(OnlyOrigin { kind: OnlyKind::Ranked { query: query.to_string(), meaning }, ids, sort: Some(sort) });
        }
        // (the warning just shown says more than "nothing found")
        Ok(_) if warned => {}
        Ok(_) => app.toast(ctx, crate::i18n::tr_format!("No photos match “{query}”", query = query)),
        Err(e) => app.toast_error(ctx, e.to_string()),
    }
}

/// Photo ▸ Show Duplicates (`photo.showDuplicates`): the grid shows photo `id` (the active one by default) and the
/// photos that are the same picture, under a "Duplicates of …" chip. None found: the view stays as it was.
pub fn show_duplicates(app: &mut LightcraftApp, ctx: &egui::Context, id: Option<PhotoId>) -> Result<Value, String> {
    sync(app);
    let id = id.or_else(|| app.session.active()).ok_or_else(|| crate::i18n::tr("No photo selected").to_string())?;
    let name = app.session.catalog.photo(id).map(|p| p.file_name.clone()).ok_or_else(|| format!("no photo {}", id.0))?;
    let before = app.session.filter.only.clone();
    // a ranked search shown before keeps the sort to go back to
    let sort = app.ui.only_origin.as_ref().and_then(|o| o.sort);
    let r = app.session.execute("photo.duplicates", &json!({"id": id.0, "filter": true})).map_err(|e| e.to_string())?;
    let found = r["photos"].as_array().map_or(0, Vec::len);
    if found == 0 {
        // the engine shows the photo alone; there is nothing to compare, so the view goes back to what it was
        let _ = app.session.execute("library.filter", &json!({"only": before}));
        let pending = r["pendingDigests"].as_u64().unwrap_or(0);
        // the `similar` tier (crops, look-alikes) needs the vector lane: say where to switch it on
        let off = meaning_off(app);
        app.toast_for(ctx, none_found(&name, pending, off), if off { 8.0 } else { crate::TOAST_SECONDS });
        if off {
            offer_settings(app);
        }
        return Ok(r);
    }
    let tiers = r["photos"].as_array().into_iter().flatten().filter_map(|p| Some((p["id"].as_u64()?, p["tier"].as_str()?.to_string()))).collect();
    note_duplicates(app, &name, sort, tiers);
    if !matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid) {
        app.ui.view = ViewMode::PhotoGrid;
    }
    Ok(r)
}

/// Remember that the filter's `only` list now holds `name`'s duplicates (it was just set by `photo.duplicates`), each
/// with its tier.
pub(crate) fn note_duplicates(app: &mut LightcraftApp, name: &str, sort: Option<Sort>, tiers: Vec<(u64, String)>) {
    let ids = app.session.filter.only.clone();
    app.ui.only_origin = Some(OnlyOrigin { kind: OnlyKind::Duplicates { name: name.to_string(), tiers }, ids, sort });
}

/// What a duplicate's cell calls its tier, in `photo.duplicates`'s order: the same bytes, the same rendered pixels (one
/// CPNP-1 Canonical Buffer: a grouping of renditions, never an identity), then the fingerprint tiers and the vector
/// lane's.
pub(crate) fn tier_label(tier: &str) -> Option<&'static str> {
    match tier {
        "bytes" => Some("Same file"),
        "pixels" => Some("Same pixels"),
        "picture" => Some("Same picture"),
        "possible" => Some("Possibly the same"),
        "similar" => Some("Looks similar"),
        _ => None,
    }
}

/// The tier of photo `id` in the duplicates the grid shows (none for the photo they are duplicates of, or when the grid
/// shows something else).
pub(crate) fn duplicate_tier(app: &LightcraftApp, id: PhotoId) -> Option<&str> {
    let origin = app.ui.only_origin.as_ref().filter(|o| o.ids == app.session.filter.only)?;
    let OnlyKind::Duplicates { tiers, .. } = &origin.kind else { return None };
    tiers.iter().find(|(t, _)| *t == id.0).map(|(_, tier)| tier.as_str())
}

/// The toast when a photo has no duplicates (yet: some photos may not have their fingerprints; and crops are found
/// only by meaning, when that is `off`).
fn none_found(name: &str, pending: u64, off: bool) -> String {
    let mut text = if pending > 0 {
        crate::i18n::tr_format!("No duplicates of {name} found yet · {pending} photos still to fingerprint", name = name, pending = pending)
    } else {
        crate::i18n::tr_format!("No duplicates of {name} found", name = name)
    };
    if off {
        text.push_str(crate::i18n::tr(" · search by meaning is off: Settings → Search"));
    }
    text
}

/// The vector lane's state (`vera.status`) and whether the session has a folder to keep its model in.
pub(crate) fn vector_state(app: &mut LightcraftApp) -> (String, String, bool) {
    let st = app.session.execute("vera.status", &json!({})).unwrap_or(Value::Null);
    let v = &st["lanes"]["vector"];
    let state = v["state"].as_str().unwrap_or("").to_string();
    let reason = v["reason"].as_str().unwrap_or("").to_string();
    (state, reason, app.session.vera_model_dir.is_some())
}

/// Whether search by meaning is off in a way Settings ▸ Search can change: switched off, or its model not downloaded
/// (and there is somewhere to keep it).
fn meaning_off(app: &mut LightcraftApp) -> bool {
    let (state, _, has_dir) = vector_state(app);
    state == "disabled_by_user" || (state == "needs_model" && has_dir)
}

/// The toast just shown gets a button that opens Settings ▸ Search.
pub(crate) fn offer_settings(app: &mut LightcraftApp) {
    offer(app, crate::i18n::tr("Open Settings → Search"), "app.settings", json!({"tab": "search"}));
}

/// The toast just shown gets a button `label` that runs `command`.
pub(crate) fn offer(app: &mut LightcraftApp, label: &str, command: &str, params: Value) {
    let Some((text, ..)) = &app.ui.toast else { return };
    app.ui.toast_action = Some(crate::state::ToastAction { text: text.clone(), label: label.to_string(), command: command.into(), params });
}

/// Photo ▸ Find Photos That Look Alike (`photo.showLookAlikes`): the grid shows photo `id` (the active one by
/// default) and the pictures nearest to it by the vector lane (`photo.similar`), best first, under a "Looks like …"
/// chip. None above the noise ceiling: the view stays as it was. The lane not ready: a toast says why, with a button
/// to Settings ▸ Search when that is where it is fixed; the command's error is returned for agents.
pub fn show_look_alikes(app: &mut LightcraftApp, ctx: &egui::Context, id: Option<PhotoId>) -> Result<Value, String> {
    sync(app);
    let id = id.or_else(|| app.session.active()).ok_or_else(|| crate::i18n::tr("No photo selected").to_string())?;
    let name = app.session.catalog.photo(id).map(|p| p.file_name.clone()).ok_or_else(|| format!("no photo {}", id.0))?;
    let sort = return_sort(app);
    let r = match app.session.execute("photo.similar", &json!({"id": id.0, "filter": true})) {
        Ok(r) => r,
        Err(e) => {
            let e = e.to_string();
            not_ready(app, ctx, &e);
            return Err(e);
        }
    };
    if r["filtered"] != true {
        app.toast(ctx, crate::i18n::tr_format!("No photos look like {name}", name = name));
        return Ok(r);
    }
    let ids = app.session.filter.only.clone();
    app.ui.only_origin = Some(OnlyOrigin { kind: OnlyKind::LooksLike { name }, ids, sort: Some(sort) });
    if !matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid) {
        app.ui.view = ViewMode::PhotoGrid;
    }
    Ok(r)
}

/// The toast when the vector lane cannot answer `photo.similar` (its `error`), in the app's words where the engine's
/// are written for agents.
fn not_ready(app: &mut LightcraftApp, ctx: &egui::Context, error: &str) {
    let (state, reason, has_dir) = vector_state(app);
    let (text, offer) = match state.as_str() {
        "needs_model" if has_dir => {
            (crate::i18n::tr("Finding photos that look alike needs the SigLIP 2 model: a one-time download, in Settings → Search.").to_string(), true)
        }
        "disabled_by_user" => {
            (crate::i18n::tr("Search by meaning is switched off for this library: Settings → Search switches it on.").to_string(), true)
        }
        "downloading" => {
            (crate::i18n::tr("The SigLIP 2 model is still downloading: Find Photos That Look Alike works once it is in place.").to_string(), false)
        }
        "live" => (super::faces::sentence(error.rsplit(": ").next().unwrap_or(error)), false),
        _ => (reason, false),
    };
    app.toast_for(ctx, text, 8.0);
    if offer {
        offer_settings(app);
    }
}

/// The sort to go back to when ranked results go: the one from before the first of a run of searches.
fn return_sort(app: &LightcraftApp) -> Sort {
    match app.ui.only_origin.as_ref().and_then(|o| o.sort) {
        Some(sort) => sort,
        // a relevance sort with nothing ranked (an agent chose it) has nothing to go back to: the default
        None if app.session.sort.key == SortKey::Relevance => Sort::default(),
        None => app.session.sort,
    }
}

/// The × on the "Ranked" / "Duplicates" chip (and typing a new search): the results go, the sort comes back.
pub fn clear_only(app: &mut LightcraftApp) {
    let _ = app.session.execute("library.filter", &json!({"only": []}));
    sync(app);
}

/// Once the filter's `only` list is no longer the one a ranked search or Show Duplicates set (cleared by the chip's
/// ×, Clear Filters, a new source, or replaced), forget where it came from and give back the sort from before the
/// search, unless the user has picked another sort meanwhile.
pub fn sync(app: &mut LightcraftApp) {
    if app.ui.only_origin.as_ref().is_none_or(|o| o.ids == app.session.filter.only) {
        return;
    }
    let Some(origin) = app.ui.only_origin.take() else { return };
    if let Some(sort) = origin.sort
        && app.session.sort.key == SortKey::Relevance
    {
        let _ = app.session.execute("library.sort", &serde_json::to_value(sort).unwrap_or_default());
    }
    // the words in the search field were the ranked results' query: they go with them
    if let OnlyKind::Ranked { query, .. } = &origin.kind
        && app.ui.search.trim() == query
        && app.session.filter.text.is_empty()
    {
        app.ui.search.clear();
    }
}

/// What the chip for the filter's `only` list says, when it came from a ranked search, Show Duplicates or Find Photos
/// That Look Alike.
pub fn only_label(app: &LightcraftApp) -> Option<String> {
    let origin = app.ui.only_origin.as_ref().filter(|o| o.ids == app.session.filter.only)?;
    Some(match &origin.kind {
        OnlyKind::Ranked { query, meaning: false } => crate::i18n::tr_format!("Ranked: “{query}”", query = query),
        OnlyKind::Ranked { query, meaning: true } => crate::i18n::tr_format!("Ranked: “{query}” · by words and meaning", query = query),
        OnlyKind::Duplicates { name, .. } => crate::i18n::tr_format!("Duplicates of {name}", name = name),
        OnlyKind::LooksLike { name } => crate::i18n::tr_format!("Looks like {name}", name = name),
    })
}

/// `ui.inspect` → `vera`: how far background fingerprinting (and embedding, when the vector lane runs) has got and the
/// pace it was last given.
pub fn inspect(app: &LightcraftApp) -> Value {
    let (pending, embedded) = app.caches.vera_embedding;
    json!({
        "pending": app.caches.vera_pending,
        "digested": app.caches.vera_digested,
        "pace": app.caches.vera_pace,
        "embedding": {"pending": pending, "embedded": embedded},
        "describe": {"pending": app.caches.vera_describe.0, "inFlight": app.caches.vera_describe.1, "proposals": app.caches.vera_describe.2},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pump_is_asked_often_only_while_there_is_work() {
        assert_eq!(schedule(12, 30), (0.05, Some(0.05)), "work left: again in 50 ms");
        assert_eq!(schedule(0, 30), (1.0, Some(5.0)), "done: a look every few seconds");
        assert_eq!(schedule(0, 0), (0.0, None), "an empty library asks for no frames");
    }

    #[test]
    fn no_duplicates_says_whether_fingerprints_are_still_coming() {
        assert_eq!(none_found("a.jpg", 0, false), "No duplicates of a.jpg found");
        assert_eq!(none_found("a.jpg", 3, false), "No duplicates of a.jpg found yet · 3 photos still to fingerprint");
        assert_eq!(none_found("a.jpg", 0, true), "No duplicates of a.jpg found · search by meaning is off: Settings → Search");
    }
}
