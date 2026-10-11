//! VERA in the app (`docs/vera.md`): fingerprinting the library in the background, the ranked search the search field
//! runs on Enter, and Show Duplicates. The engine does the work (`vera.pump`, `search.text`, `photo.duplicates`); this
//! module decides when to call it, names what the grid shows in its chip ("Ranked: “dog”", "Duplicates of
//! IMG_0042.JPG") and puts the view back when those results go.

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
    if now < app.caches.vera_next_pump {
        return;
    }
    let (pace, _) = super::faces::scan_pace(app, ctx, now);
    let Ok(v) = app.session.execute("vera.pump", &json!({"pace": pace})) else { return };
    app.caches.vera_pace = pace;
    app.caches.vera_pending = v["pending"].as_u64().unwrap_or(0);
    app.caches.vera_digested = v["digested"].as_u64().unwrap_or(0);
    let photos = v["photos"].as_u64().unwrap_or(0);
    // as for the face scan: called again in 50 ms while there is work, looked at again every few seconds once it is
    // done (new photos are fingerprinted soon after they arrive), and nothing asked for while the library is empty
    let (next, wait) = schedule(app.caches.vera_pending, photos);
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
    match app.session.execute("search.text", &json!({"query": query})) {
        Ok(r) if r["filtered"] == true => {
            // the live filter from typing would narrow the ranked list: the ranking replaces it (the field keeps the words)
            if !app.session.filter.text.is_empty() {
                let _ = app.session.execute("library.filter", &json!({"text": ""}));
            }
            let ids = app.session.filter.only.clone();
            // the photo shown was left out: the best match takes its place (in the loupe, and as the grid's active photo)
            if let Some(best) = ids.first().filter(|_| app.session.active().is_none_or(|a| !ids.contains(&a))) {
                let _ = app.session.execute("library.select", &json!({"ids": [best.0]}));
            }
            app.ui.only_origin = Some(OnlyOrigin { kind: OnlyKind::Ranked { query: query.to_string() }, ids, sort: Some(sort) });
        }
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
        app.toast(ctx, none_found(&name, pending));
        return Ok(r);
    }
    note_duplicates(app, &name, sort);
    if !matches!(app.ui.view, ViewMode::PhotoGrid | ViewMode::SquareGrid) {
        app.ui.view = ViewMode::PhotoGrid;
    }
    Ok(r)
}

/// Remember that the filter's `only` list now holds `name`'s duplicates (it was just set by `photo.duplicates`).
pub(crate) fn note_duplicates(app: &mut LightcraftApp, name: &str, sort: Option<Sort>) {
    let ids = app.session.filter.only.clone();
    app.ui.only_origin = Some(OnlyOrigin { kind: OnlyKind::Duplicates { name: name.to_string() }, ids, sort });
}

/// The toast when a photo has no duplicates (yet: some photos may not have their fingerprints).
fn none_found(name: &str, pending: u64) -> String {
    if pending > 0 {
        crate::i18n::tr_format!("No duplicates of {name} found yet · {pending} photos still to fingerprint", name = name, pending = pending)
    } else {
        crate::i18n::tr_format!("No duplicates of {name} found", name = name)
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
    if let OnlyKind::Ranked { query } = &origin.kind
        && app.ui.search.trim() == query
        && app.session.filter.text.is_empty()
    {
        app.ui.search.clear();
    }
}

/// What the chip for the filter's `only` list says, when it came from a ranked search or Show Duplicates.
pub fn only_label(app: &LightcraftApp) -> Option<String> {
    let origin = app.ui.only_origin.as_ref().filter(|o| o.ids == app.session.filter.only)?;
    Some(match &origin.kind {
        OnlyKind::Ranked { query } => crate::i18n::tr_format!("Ranked: “{query}”", query = query),
        OnlyKind::Duplicates { name } => crate::i18n::tr_format!("Duplicates of {name}", name = name),
    })
}

/// `ui.inspect` → `vera`: how far background fingerprinting has got and the pace it was last given.
pub fn inspect(app: &LightcraftApp) -> Value {
    json!({"pending": app.caches.vera_pending, "digested": app.caches.vera_digested, "pace": app.caches.vera_pace})
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
        assert_eq!(none_found("a.jpg", 0), "No duplicates of a.jpg found");
        assert_eq!(none_found("a.jpg", 3), "No duplicates of a.jpg found yet · 3 photos still to fingerprint");
    }
}
