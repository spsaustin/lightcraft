//! Describe in the app (`docs/vera.md` → Phase 3): Photo ▸ Describe Photos… and Suggest Keywords, which queue the
//! selection for the engine's Describe lane (`photo.describe`), and Window ▸ Review Descriptions, where its proposals
//! (`describe.list`) are read one at a time: each field beside the photo's own value, editable in place and ticked to
//! keep, then Accept (`describe.accept`, one undo step), Reject (`describe.reject`), Skip or Reject All
//! (`describe.clear`). The engine does the work; nothing is written to a photo but by Accept. The endpoint is set in
//! Settings ▸ Search (`panels::vera_settings`), and the Info panel marks a field that came from here (`ai_tag`).

use egui::RichText;
use lightcraft_catalog::Provenance;
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::state::{DescribeReview, Dialog};
use crate::theme::Tokens;
use crate::widgets::register;

/// The fields a proposal can hold, in the order the review shows them: (`describe.accept`'s name, label).
pub const FIELDS: [(&str, &str); 4] = [("title", "Title"), ("caption", "Caption"), ("alt_text", "Alt Text"), ("keywords", "Keywords")];

/// How long a read of the queue is reused while the review is open (new proposals arrive while a run goes on).
const REFRESH_SECS: f64 = 0.5;

/// The SigLIP 2 model id the engine names local keyword proposals by.
const LOCAL_MODEL: &str = "siglip2";

fn cache_id() -> egui::Id {
    egui::Id::new("describe-review-list")
}

/// Photo ▸ Describe Photos… (`keywords_only: false`) and Photo ▸ Suggest Keywords (`true`): queue the selection (or
/// the active photo) for Describe. The activity stack shows the run ("Describing photos"); its proposals wait for
/// review. When the lane cannot run, a toast says why in the app's words, with a button to Settings ▸ Search where
/// that is where it is fixed; the command's error is returned for agents.
pub fn describe_selected(app: &mut LightcraftApp, ctx: &egui::Context, keywords_only: bool) -> Result<Value, String> {
    let ids: Vec<u64> = app.session.targets(&json!({})).into_iter().map(|id| id.0).collect();
    if ids.is_empty() {
        return Err(crate::i18n::tr("No photo selected").to_string());
    }
    let mut params = json!({"ids": ids});
    if keywords_only {
        params["keywordsOnly"] = json!(true);
    }
    let r = match app.session.execute("photo.describe", &params) {
        Ok(r) => r,
        Err(e) => {
            let e = e.to_string();
            not_ready(app, ctx, keywords_only, &e);
            return Err(e);
        }
    };
    let queued = r["queued"].as_u64().unwrap_or(0);
    let text = match (queued, keywords_only) {
        (0, _) => crate::i18n::tr("These photos are being described already").to_string(),
        (n, false) => crate::i18n::tr_format!(
            "Describing {n} photos with {host}: each description waits for your review",
            n = n,
            host = r["host"].as_str().unwrap_or("")
        ),
        (n, true) => crate::i18n::tr_format!("Suggesting keywords for {n} photos: they wait for your review", n = n),
    };
    app.toast(ctx, text);
    Ok(r)
}

/// The toast when `photo.describe` refused (its `error`): what is missing, and where it is set.
fn not_ready(app: &mut LightcraftApp, ctx: &egui::Context, keywords_only: bool, error: &str) {
    let (text, offer) = if keywords_only {
        let (state, reason, has_dir) = super::vera::vector_state(app);
        match state.as_str() {
            "needs_model" if has_dir => {
                (crate::i18n::tr("Suggesting keywords needs the SigLIP 2 model: a one-time download, in Settings → Search.").to_string(), true)
            }
            "disabled_by_user" => {
                (crate::i18n::tr("Search by meaning is switched off for this library: Settings → Search switches it on.").to_string(), true)
            }
            "downloading" => {
                (crate::i18n::tr("The SigLIP 2 model is still downloading: Suggest Keywords works once it is in place.").to_string(), false)
            }
            "live" => (super::faces::sentence(super::vera_settings::engine_message(error)), false),
            _ => (reason, false),
        }
    } else {
        let st = app.session.execute("vera.status", &json!({})).unwrap_or(Value::Null);
        let d = &st["lanes"]["describe"];
        match d["state"].as_str().unwrap_or("") {
            "needs_endpoint" => (
                crate::i18n::tr("Describing photos needs an endpoint: a llama.cpp, Ollama, vLLM or LM Studio server, set in Settings → Search.")
                    .to_string(),
                true,
            ),
            "needs_token" => (
                crate::i18n::tr_format!("The endpoint at {host} needs an access token: Settings → Search.", host = d["host"].as_str().unwrap_or("")),
                true,
            ),
            "error" => (d["reason"].as_str().unwrap_or(error).to_string(), true),
            "live" => (super::faces::sentence(super::vera_settings::engine_message(error)), false),
            _ => (d["reason"].as_str().unwrap_or(error).to_string(), false),
        }
    };
    app.toast_for(ctx, text, 8.0);
    if offer {
        super::vera::offer_settings(app);
    }
}

/// A Describe run has ended (nothing waiting or in flight): say how many proposals wait, with a button to review them.
pub fn run_ended(app: &mut LightcraftApp, ctx: &egui::Context) {
    let proposals = app.caches.vera_describe.2;
    if proposals == 0 || matches!(app.ui.dialog, Some(Dialog::ReviewDescriptions { .. })) {
        return;
    }
    app.toast_for(ctx, crate::i18n::tr_format!("{n} descriptions wait for your review", n = proposals), 10.0);
    super::vera::offer(app, crate::i18n::tr("Review"), "view.reviewDescriptions", json!({}));
}

/// Window ▸ Review Descriptions (`view.reviewDescriptions`).
pub fn open(app: &mut LightcraftApp, ctx: &egui::Context) -> Result<Value, String> {
    // the toast that offered the review has done its job
    if app.ui.toast_action.as_ref().is_some_and(|a| a.command == "view.reviewDescriptions") {
        app.ui.toast = None;
        app.ui.toast_action = None;
    }
    ctx.data_mut(|d| d.remove::<(f64, (u64, u64, u64), Value)>(cache_id()));
    if !matches!(app.ui.dialog, Some(Dialog::ReviewDescriptions { .. })) {
        app.ui.dialog = Some(Dialog::ReviewDescriptions { review: DescribeReview::default() });
    }
    let l = app.session.execute("describe.list", &json!({"limit": 1})).map_err(|e| e.to_string())?;
    Ok(json!({"proposals": l["total"], "pending": l["pending"]}))
}

/// The queue: enough proposals to find the first one not skipped, with the counts (re-read every [`REFRESH_SECS`], and
/// as soon as the pump reports a change in Describe's counts).
fn read(app: &mut LightcraftApp, ctx: &egui::Context, skipped: usize) -> Value {
    let now = ctx.input(|i| i.time);
    let counts = app.caches.vera_describe;
    if let Some((at, seen, list)) = ctx.data(|d| d.get_temp::<(f64, (u64, u64, u64), Value)>(cache_id()))
        && now - at < REFRESH_SECS
        && seen == counts
    {
        return list;
    }
    let limit = skipped.saturating_add(1).min(10_000);
    let list = app.session.execute("describe.list", &json!({"limit": limit})).unwrap_or(Value::Null);
    ctx.data_mut(|d| d.insert_temp(cache_id(), (now, counts, list.clone())));
    // the run brings new proposals: look again
    ctx.request_repaint_after(std::time::Duration::from_secs_f64(REFRESH_SECS));
    list
}

/// Forget the cached queue (after an action changed it).
fn reread(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove::<(f64, (u64, u64, u64), Value)>(cache_id()));
}

/// The proposal to show: the first not skipped; once every proposal was skipped, the round starts again.
fn current(list: &Value, review: &mut DescribeReview) -> Option<Value> {
    let rows = list["proposals"].as_array()?;
    let first = |skipped: &[u64]| rows.iter().find(|p| p["photo"].as_u64().is_some_and(|id| !skipped.contains(&id))).cloned();
    if let Some(p) = first(&review.skipped) {
        return Some(p);
    }
    review.skipped.clear();
    first(&[])
}

/// Fill the review's fields from proposal `p` (when it is another photo's than the one they hold).
fn load(review: &mut DescribeReview, p: &Value) {
    let photo = p["photo"].as_u64();
    if review.photo == photo {
        return;
    }
    let f = &p["fields"];
    let text = |k: &str| f[k].as_str().unwrap_or("").to_string();
    review.photo = photo;
    review.title = text("title");
    review.caption = text("caption");
    review.alt_text = text("alt_text");
    review.keywords = f["keywords"].as_array().into_iter().flatten().filter_map(|k| k.as_str().map(str::to_string)).collect();
    review.new_keyword.clear();
    // a field with no proposal is absent; every other one is ticked
    review.fields = FIELDS
        .iter()
        .filter(|(k, _)| match f.get(*k) {
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            _ => false,
        })
        .map(|(k, _)| ((*k).to_string(), true))
        .collect();
    review.error = None;
}

/// The value of `field` as the review holds it.
fn edited(review: &DescribeReview, field: &str) -> Value {
    match field {
        "title" => json!(review.title.trim()),
        "caption" => json!(review.caption.trim()),
        "alt_text" => json!(review.alt_text.trim()),
        _ => json!(review.keywords),
    }
}

/// What Accept sends: the ticked fields that still hold something, the edits where the text differs from the
/// proposal, and whether the keywords replace the photo's.
pub(crate) fn accept_params(review: &DescribeReview, proposal: &Value) -> Option<Value> {
    let id = review.photo?;
    let mut fields = Vec::new();
    let mut edits = serde_json::Map::new();
    for (field, ticked) in &review.fields {
        let value = edited(review, field);
        // an emptied field has nothing to accept
        let empty = value.as_str().is_some_and(str::is_empty) || value.as_array().is_some_and(Vec::is_empty);
        if !ticked || empty {
            continue;
        }
        fields.push(field.clone());
        if proposal["fields"][field.as_str()] != value {
            edits.insert(field.clone(), value);
        }
    }
    if fields.is_empty() {
        return None;
    }
    let mut p = json!({"id": id, "fields": fields});
    if !edits.is_empty() {
        p["edits"] = Value::Object(edits);
    }
    if review.replace_keywords && fields.iter().any(|f| f == "keywords") {
        p["replaceKeywords"] = json!(true);
    }
    Some(p)
}

/// Who wrote a proposal: "Proposed by <model> at <host>", or the local keywords' line.
pub(crate) fn source_line(source: &Value) -> String {
    let model = source["model"].as_str().unwrap_or("");
    match (source["kind"].as_str(), source["host"].as_str()) {
        (Some("local"), _) => crate::i18n::tr("Keywords from SigLIP 2 on this computer").to_string(),
        (_, Some(host)) => crate::i18n::tr_format!("Proposed by {model} at {host}", model = model, host = host),
        _ => crate::i18n::tr_format!("Proposed by {model}", model = model),
    }
}

/// What the review does on this frame (from its buttons or keys).
enum Act {
    Accept,
    Reject,
    Skip,
    Clear,
}

/// The review's body (inside the dialog frame, whose Close button and Esc close it).
pub fn body(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, review: &mut DescribeReview) {
    let ctx = ui.ctx().clone();
    // keys first, while a text field that had the focus still has it (Return there ends the edit, not the review)
    let mut act = None;
    if !ctx.egui_wants_keyboard_input() && !review.confirm_clear {
        ui.input_mut(|i| {
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Enter) {
                act = Some(Act::Accept);
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace) || i.consume_key(egui::Modifiers::NONE, egui::Key::Delete) {
                act = Some(Act::Reject);
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight) {
                act = Some(Act::Skip);
            }
        });
    }
    let list = read(app, &ctx, review.skipped.len());
    let total = list["total"].as_u64().unwrap_or(0);
    let waiting = list["pending"].as_u64().unwrap_or(0).saturating_add(list["inFlight"].as_u64().unwrap_or(0));
    let proposal = current(&list, review);
    match &proposal {
        Some(p) => load(review, p),
        None => {
            review.photo = None;
            act = None;
            let text = if waiting > 0 {
                crate::i18n::tr_format!("Nothing to review yet · {n} photos are being described", n = waiting)
            } else {
                crate::i18n::tr("Nothing to review. Photo → Describe Photos… proposes words for the selected photos.").to_string()
            };
            let r = ui.add(egui::Label::new(RichText::new(text).color(t.text_label)).wrap());
            register(ui.ctx(), "describe:empty", r.rect);
        }
    }
    if let Some(p) = &proposal {
        proposal_view(app, ui, t, review, p, total, &mut act);
    }
    footer(ui, t, review, total, waiting);
    let Some(act) = act else { return };
    let Some(id) = review.photo else { return };
    match act {
        Act::Accept => {
            let Some(p) = proposal.as_ref().and_then(|p| accept_params(review, p)) else {
                review.error = Some(crate::i18n::tr("Tick a field that has words to accept, or Reject the proposal").to_string());
                return;
            };
            match app.run("describe.accept", p) {
                Ok(_) => {
                    review.accepted += 1;
                    review.photo = None;
                    reread(&ctx);
                }
                Err(e) => review.error = Some(e),
            }
        }
        Act::Reject => match app.run("describe.reject", json!({"id": id})) {
            Ok(_) => {
                review.rejected += 1;
                review.photo = None;
                reread(&ctx);
            }
            Err(e) => review.error = Some(e),
        },
        Act::Skip => {
            review.skipped.push(id);
            review.photo = None;
            reread(&ctx);
        }
        Act::Clear => match app.run("describe.clear", json!({})) {
            Ok(r) => {
                review.rejected += usize::try_from(r["removed"].as_u64().unwrap_or(0)).unwrap_or(0);
                review.photo = None;
                review.skipped.clear();
                review.confirm_clear = false;
                reread(&ctx);
            }
            Err(e) => review.error = Some(e),
        },
    }
}

/// One proposal: the photo and who wrote it, each field beside the photo's own value, and the buttons.
fn proposal_view(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, review: &mut DescribeReview, p: &Value, total: u64, act: &mut Option<Act>) {
    let Some(id) = p["photo"].as_u64() else { return };
    let photo = lightcraft_catalog::PhotoId(id);
    let name = app.session.catalog.photo(photo).map(|ph| ph.file_name.clone()).unwrap_or_default();
    ui.horizontal(|ui| {
        thumbnail(app, ui, t, photo);
        ui.vertical(|ui| {
            let r = ui.label(RichText::new(&name).font(t.semibold(13.5)).color(t.text));
            register(ui.ctx(), "describe:photo", r.rect);
            let r = ui.label(RichText::new(source_line(&p["source"])).font(t.font(11.5)).color(t.text_dim));
            register(ui.ctx(), "describe:source", r.rect);
            let left = crate::i18n::tr_format!("{n} to review", n = total);
            ui.label(RichText::new(left).font(t.font(11.5)).color(t.text_dim));
        });
    });
    ui.add_space(2.0);
    let current = &p["current"];
    egui::Grid::new("describe-review-fields").num_columns(3).spacing([12.0, 8.0]).min_col_width(90.0).show(ui, |ui| {
        ui.label("");
        ui.label(RichText::new(crate::i18n::tr("Now")).font(t.font(11.5)).color(t.text_dim));
        ui.label(RichText::new(crate::i18n::tr("Proposed")).font(t.font(11.5)).color(t.text_dim));
        ui.end_row();
        let fields = review.fields.clone();
        for (i, (field, ticked)) in fields.iter().enumerate() {
            let label = FIELDS.iter().find(|(k, _)| k == field).map_or(field.as_str(), |(_, l)| *l);
            let mut on = *ticked;
            let r = ui.checkbox(&mut on, crate::i18n::tr(label));
            register(ui.ctx(), format!("check:describe.{field}"), r.rect);
            if r.changed()
                && let Some(f) = review.fields.get_mut(i)
            {
                f.1 = on;
            }
            let now = match &current[field.as_str()] {
                Value::Array(a) => a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "),
                v => v.as_str().unwrap_or("").to_string(),
            };
            let shown = if now.is_empty() { "—".to_string() } else { now };
            ui.vertical(|ui| {
                ui.set_width(200.0);
                ui.add(egui::Label::new(RichText::new(shown).font(t.font(12.0)).color(t.text_label)).wrap());
            });
            // an unticked field is kept as it is, greyed out
            ui.vertical(|ui| {
                ui.add_enabled_ui(on, |ui| {
                    ui.set_width(330.0);
                    let widget = format!("field:describe.{field}");
                    match field.as_str() {
                        "title" => {
                            crate::text_field::TextField::singleline(&widget, &mut review.title).width(330.0).show(ui);
                        }
                        "caption" => {
                            crate::text_field::TextField::multiline(&widget, &mut review.caption).rows(2).width(330.0).show(ui);
                        }
                        "alt_text" => {
                            crate::text_field::TextField::multiline(&widget, &mut review.alt_text).rows(2).width(330.0).show(ui);
                        }
                        _ => keywords(ui, review),
                    }
                });
            });
            ui.end_row();
        }
    });
    if review.fields.iter().any(|(f, _)| f == "keywords") {
        let r = ui.checkbox(&mut review.replace_keywords, crate::i18n::tr("Replace the photo's keywords (otherwise they are added)"));
        register(ui.ctx(), "check:describe.replaceKeywords", r.rect);
    }
    if let Some(e) = &review.error {
        let r = ui.add(egui::Label::new(RichText::new(e).font(t.font(11.5)).color(t.caution)).wrap());
        register(ui.ctx(), "describe:error", r.rect);
    }
    ui.add_space(2.0);
    if review.confirm_clear {
        let text = crate::i18n::tr_format!("Reject all {n} proposals? Nothing is written to the photos.", n = total);
        ui.label(RichText::new(text).color(t.text_label));
        ui.horizontal(|ui| {
            let keep = ui.button(crate::i18n::tr("Keep"));
            register(ui.ctx(), "button:describeRejectAllCancel", keep.rect);
            let all = ui.button(crate::i18n::tr("Reject All"));
            register(ui.ctx(), "button:describeRejectAllConfirm", all.rect);
            if keep.clicked() {
                review.confirm_clear = false;
            }
            if all.clicked() {
                *act = Some(Act::Clear);
            }
        });
        return;
    }
    ui.horizontal(|ui| {
        let can = accept_params(review, p).is_some();
        let r = ui.add_enabled(can, egui::Button::new(RichText::new(crate::i18n::tr("Accept")).strong()));
        register(ui.ctx(), "button:describeAccept", r.rect);
        if r.on_hover_text(crate::i18n::tr("Write the ticked fields to the photo (Return)")).clicked() {
            *act = Some(Act::Accept);
        }
        let r = ui.button(crate::i18n::tr("Reject"));
        register(ui.ctx(), "button:describeReject", r.rect);
        if r.on_hover_text(crate::i18n::tr("Drop this proposal (⌫)")).clicked() {
            *act = Some(Act::Reject);
        }
        let r = ui.button(crate::i18n::tr("Skip"));
        register(ui.ctx(), "button:describeSkip", r.rect);
        if r.on_hover_text(crate::i18n::tr("Next proposal; this one stays (→)")).clicked() {
            *act = Some(Act::Skip);
        }
        ui.add_space(16.0);
        let r = ui.button(crate::i18n::tr("Reject All…"));
        register(ui.ctx(), "button:describeRejectAll", r.rect);
        if r.clicked() {
            review.confirm_clear = true;
        }
    });
    ui.label(RichText::new(crate::i18n::tr("Return accepts · ⌫ rejects · → skips")).font(t.font(11.0)).color(t.text_dim));
}

/// The proposed keywords as chips (× drops one) and a field to add one.
fn keywords(ui: &mut egui::Ui, review: &mut DescribeReview) {
    let mut drop = None;
    ui.horizontal_wrapped(|ui| {
        for (i, k) in review.keywords.iter().enumerate() {
            let (body, x) = super::keywording::chip(ui, k, false, true);
            register(ui.ctx(), format!("describe:keyword:{k}"), body.rect);
            if let Some(x) = x {
                register(ui.ctx(), format!("describe:dropKeyword:{k}"), x.rect);
                if x.on_hover_text(crate::i18n::tr("Leave this keyword out")).clicked() {
                    drop = Some(i);
                }
            }
        }
    });
    if let Some(i) = drop
        && i < review.keywords.len()
    {
        review.keywords.remove(i);
    }
    let r = crate::text_field::TextField::singleline("field:describe.addKeyword", &mut review.new_keyword)
        .hint(crate::i18n::tr("Add keyword"))
        .width(330.0)
        .show(ui);
    if r.ending == Some(crate::text_field::Ending::Return) {
        for k in review.new_keyword.split(',').map(str::trim).filter(|k| !k.is_empty()) {
            if !review.keywords.iter().any(|x| x.eq_ignore_ascii_case(k)) {
                review.keywords.push(k.to_string());
            }
        }
        review.new_keyword.clear();
    }
}

/// The photo's thumbnail, in a box of its own.
fn thumbnail(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, id: lightcraft_catalog::PhotoId) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(180.0, 135.0), egui::Sense::hover());
    register(ui.ctx(), "describe:thumbnail", rect);
    ui.painter().rect_filled(rect, 4.0, t.canvas);
    // the grid behind usually has it; asked for only when it has none (two sizes asked for one photo would replace
    // each other's render without end)
    if app.renderer.thumb(id).is_none() {
        super::grid::request_thumb(app, id, 256, 8);
    }
    if let Some(tex) = app.renderer.thumb(id) {
        let [tw, th] = tex.size;
        let s = (rect.width() / tw.max(1) as f32).min(rect.height() / th.max(1) as f32);
        let fr = egui::Rect::from_center_size(rect.center(), egui::vec2(tw as f32 * s, th as f32 * s));
        ui.painter().image(tex.tex.id(), fr, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    } else {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
}

/// This session's counts and the queue's.
fn footer(ui: &mut egui::Ui, t: &Tokens, review: &DescribeReview, total: u64, waiting: u64) {
    ui.separator();
    let text = crate::i18n::tr_format!(
        "This session: {accepted} accepted · {rejected} rejected · {waiting} being described · {total} to review",
        accepted = review.accepted,
        rejected = review.rejected,
        waiting = waiting,
        total = total
    );
    let r = ui.label(RichText::new(text).font(t.font(11.5)).color(t.text_dim));
    register(ui.ctx(), "describe:counts", r.rect);
}

/// The small "AI" tag beside a field whose value Describe proposed and the user accepted (`provenance` of the
/// field), naming the model on hover. Nothing for any other origin.
pub fn ai_tag(ui: &mut egui::Ui, t: &Tokens, widget: &str, provenance: Option<&Provenance>) {
    let Some(Provenance::Describe { model, at }) = provenance else { return };
    let r = egui::Frame::NONE
        .fill(t.chrome)
        .stroke(egui::Stroke::new(1.0, t.field_border))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::symmetric(4, 0))
        .show(ui, |ui| ui.label(RichText::new("AI").font(t.semibold(9.5)).color(t.text_label)))
        .response;
    register(ui.ctx(), format!("ai:{widget}"), r.rect);
    let who = if model.starts_with(LOCAL_MODEL) {
        crate::i18n::tr("Keywords from SigLIP 2 on this computer").to_string()
    } else {
        crate::i18n::tr_format!("Written by {model}", model = model)
    };
    let when = at.get(..10).unwrap_or(at);
    r.on_hover_text(crate::i18n::tr_format!("{who}, accepted {when}", who = who, when = when));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn review_of(p: &Value) -> DescribeReview {
        let mut r = DescribeReview::default();
        load(&mut r, p);
        r
    }

    #[test]
    fn accept_sends_the_ticked_fields_and_only_the_edits() {
        let p = json!({"photo": 7, "fields": {"title": "Lake", "alt_text": "A lake.", "keywords": ["lake", "dawn"]}, "source": {"kind": "remote", "model": "m", "host": "h:1"}});
        let mut r = review_of(&p);
        assert_eq!(
            r.fields,
            [("title".to_string(), true), ("alt_text".to_string(), true), ("keywords".to_string(), true)],
            "no caption proposed: absent"
        );
        assert_eq!(accept_params(&r, &p), Some(json!({"id": 7, "fields": ["title", "alt_text", "keywords"]})), "nothing edited");
        r.title = "My Lake".into();
        r.fields[1].1 = false;
        r.keywords.retain(|k| k != "dawn");
        r.replace_keywords = true;
        assert_eq!(
            accept_params(&r, &p),
            Some(json!({"id": 7, "fields": ["title", "keywords"], "edits": {"title": "My Lake", "keywords": ["lake"]}, "replaceKeywords": true}))
        );
        // emptied or unticked: nothing to accept
        r.title.clear();
        r.keywords.clear();
        assert_eq!(accept_params(&r, &p), None);
        // the same photo again keeps the edits; another photo's proposal starts afresh
        load(&mut r, &p);
        assert!(r.title.is_empty());
        load(&mut r, &json!({"photo": 8, "fields": {"title": "B"}}));
        assert_eq!((r.title.as_str(), r.fields.len()), ("B", 1));
    }

    #[test]
    fn skipped_proposals_come_round_again() {
        let list = json!({"proposals": [{"photo": 1}, {"photo": 2}]});
        let mut r = DescribeReview { skipped: vec![1], ..Default::default() };
        assert_eq!(current(&list, &mut r).unwrap()["photo"], 2);
        r.skipped.push(2);
        assert_eq!(current(&list, &mut r).unwrap()["photo"], 1, "all skipped: from the start");
        assert!(r.skipped.is_empty());
        assert!(current(&json!({"proposals": []}), &mut r).is_none());
    }

    #[test]
    fn the_source_line_names_the_model_and_where_it_ran() {
        assert_eq!(source_line(&json!({"kind": "remote", "model": "llava", "host": "127.0.0.1:8080"})), "Proposed by llava at 127.0.0.1:8080");
        assert_eq!(source_line(&json!({"kind": "local", "model": "siglip2-base-patch32-256"})), "Keywords from SigLIP 2 on this computer");
    }
}
