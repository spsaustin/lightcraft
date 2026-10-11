//! Settings ▸ Search (`docs/vera.md`): the search lanes as `vera.status` reports them, how far fingerprinting and
//! embedding have got, and the SigLIP 2 model behind search by meaning — its download (the licence first, in a dialog
//! of its own), its progress, the on/off switch and Remove. The engine does the work (`vera.model.*`, `vera.config`);
//! this module reads it at most once a second while the tab is open, and follows a download the user started until it
//! ends (`watch`, called from `panels::vera::pump`).

use egui::RichText;
use serde_json::{Value, json};

use super::settings::{check, heading, hint};
use crate::LightcraftApp;
use crate::theme::Tokens;
use crate::widgets::register;

/// How long a read of `vera.status` and `vera.model.status` is reused while the tab is open.
const REFRESH_SECS: f64 = 1.0;

/// The model download's licence text in full (the notice says the Apache-2.0 text follows).
const APACHE_URL: &str = "https://www.apache.org/licenses/LICENSE-2.0";

fn cache_id() -> egui::Id {
    egui::Id::new("vera-settings-status")
}

/// `vera.status` and `vera.model.status`, re-read at most every [`REFRESH_SECS`], or at once after an action.
fn read(app: &mut LightcraftApp, ctx: &egui::Context) -> (Value, Value) {
    let now = ctx.input(|i| i.time);
    let epoch = app.caches.vera_epoch;
    if let Some((e, at, status, model)) = ctx.data(|d| d.get_temp::<(u64, f64, Value, Value)>(cache_id()))
        && e == epoch
        && now - at < REFRESH_SECS
    {
        return (status, model);
    }
    let status = app.session.execute("vera.status", &json!({})).unwrap_or(Value::Null);
    let model = app.session.execute("vera.model.status", &json!({})).unwrap_or(Value::Null);
    ctx.data_mut(|d| d.insert_temp(cache_id(), (epoch, now, status.clone(), model.clone())));
    (status, model)
}

/// `1_511_714_589` → `1.5 GB`.
fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// What the row of a lane says about its state (the engine's state names).
pub(crate) fn state_label(state: &str) -> &'static str {
    match state {
        "live" => "On",
        "needs_model" => "Needs its model",
        "downloading" => "Downloading",
        "disabled_by_user" | "disabled" => "Off",
        "needs_endpoint" => "Needs an endpoint",
        "needs_token" => "Needs a token",
        "needs_desktop" => "Desktop app only",
        "error" => "Not working",
        _ => "Not available yet",
    }
}

/// The sentence for the vector lane in `state`. The engine's reasons are written for agents (they name the commands
/// that change them); the app says the same in its own words, and passes the engine's on where it carries a message
/// (a model that failed) or nothing else applies (no folder to keep the model in, the web build).
pub(crate) fn vector_reason(state: &str, engine_reason: &str, has_dir: bool) -> String {
    let own = match state {
        "live" => "Finds photos by what they show and pictures that look alike, with SigLIP 2 on this computer.",
        "needs_model" if has_dir => {
            "Finds photos by what they show and pictures that look alike once its model is here: a one-time download (below)."
        }
        "downloading" => "The model is downloading; search by meaning starts as soon as it is in place.",
        "disabled_by_user" => "Switched off for this library: searches use words and fingerprints only.",
        _ => return engine_reason.to_string(),
    };
    crate::i18n::tr(own).to_string()
}

/// The Settings ▸ Search tab.
pub fn settings_tab(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let (status, model) = read(app, ui.ctx());
    // the progress and the counts move on their own: look again in a second
    ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(REFRESH_SECS));
    heading(ui, t, "Search");
    hint(ui, t, "Enter in the search field ranks photos by every lane that is on. Everything runs on this computer.");
    let photos = status["photos"].as_u64().unwrap_or(0);
    let vector = &status["lanes"]["vector"];
    let vector_state = vector["state"].as_str().unwrap_or("needs_model");
    let has_dir = model["dir"].is_string();
    let lanes: [(&str, &str, &str, String); 4] = [
        (
            "keyword",
            "By words",
            status["lanes"]["keyword"].as_str().unwrap_or("live"),
            crate::i18n::tr("Title, keywords, caption, alt text, file name, camera, lens and place.").to_string(),
        ),
        (
            "digest",
            "Same picture",
            status["lanes"]["digest"].as_str().unwrap_or("live"),
            crate::i18n::tr("Resizes, re-exports and format changes of one picture, by fingerprint (Show Duplicates).").to_string(),
        ),
        ("vector", "By meaning", vector_state, vector_reason(vector_state, vector["reason"].as_str().unwrap_or(""), has_dir)),
        (
            "describe",
            "Describe",
            status["lanes"]["describe"]["state"].as_str().unwrap_or("disabled"),
            match status["lanes"]["describe"]["state"].as_str() {
                // a describer that failed says why
                Some("error") => status["lanes"]["describe"]["reason"].as_str().unwrap_or("").to_string(),
                _ => crate::i18n::tr("Words written for photos that have none, so search by words finds them.").to_string(),
            },
        ),
    ];
    ui.add_space(2.0);
    egui::Grid::new("vera-lanes").num_columns(3).spacing([14.0, 6.0]).show(ui, |ui| {
        for (id, name, state, reason) in &lanes {
            let r = ui.label(RichText::new(crate::i18n::tr(name)).color(t.text));
            register(ui.ctx(), format!("vera:lane:{id}"), r.rect);
            let colour = match *state {
                "live" => t.accent,
                "error" => t.caution,
                _ => t.text_dim,
            };
            ui.label(RichText::new(crate::i18n::tr(state_label(state))).font(t.semibold(11.5)).color(colour));
            ui.add(egui::Label::new(RichText::new(reason).font(t.font(11.5)).color(t.text_dim)).wrap());
            ui.end_row();
        }
    });
    ui.add_space(2.0);
    let digested = status["digested"].as_u64().unwrap_or(0);
    let embedded = status["embedded"].as_u64().unwrap_or(0);
    let counts = if model["installed"] == true {
        crate::i18n::tr_format!(
            "{digested} of {photos} photos fingerprinted · {embedded} of {photos} embedded",
            digested = digested,
            embedded = embedded,
            photos = photos
        )
    } else {
        crate::i18n::tr_format!("{digested} of {photos} photos fingerprinted", digested = digested, photos = photos)
    };
    let r = ui.label(RichText::new(counts).font(t.font(11.5)).color(t.text_label));
    register(ui.ctx(), "vera:counts", r.rect);
    model_section(app, ui, t, &status, &model);
}

/// The SigLIP 2 model: download (with its progress and Cancel), or the on/off switch, its size and Remove.
fn model_section(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &Value, model: &Value) {
    heading(ui, t, "Search by meaning");
    if model["state"] == "needs_desktop" {
        hint(ui, t, "Search by meaning runs in the desktop app.");
        return;
    }
    let bytes = model["bytes"].as_u64().unwrap_or(0);
    ui.horizontal(|ui| {
        ui.label(RichText::new("SigLIP 2 Base").color(t.text));
        ui.label(RichText::new(gb(bytes)).color(t.text_dim));
        ui.label(RichText::new(model["licence"]["name"].as_str().unwrap_or("Apache License 2.0")).color(t.text_dim));
        if model["installed"] == true {
            ui.label(RichText::new(crate::i18n::tr("installed")).color(t.accent));
        }
    });
    if model["installed"] == true {
        installed_rows(app, ui, t, status, model);
        return;
    }
    if let Some(d) = model["downloading"].as_object() {
        let done = d.get("done").and_then(Value::as_u64).unwrap_or(0);
        let total = d.get("total").and_then(Value::as_u64).unwrap_or(bytes).max(1);
        let file = d.get("file").and_then(Value::as_str).unwrap_or("");
        let text = crate::i18n::tr_format!("{done:.2} of {total:.2} GB · {file}", done = done as f64 / 1e9, total = total as f64 / 1e9, file = file);
        ui.horizontal(|ui| {
            let bar = ui.add(
                egui::ProgressBar::new((done as f32 / total as f32).clamp(0.0, 1.0))
                    .desired_width(380.0)
                    .text(RichText::new(text).font(t.font(11.5))),
            );
            register(ui.ctx(), "progress:veraDownload", bar.rect);
            let r = ui.button(crate::i18n::tr("Cancel"));
            register(ui.ctx(), "button:veraCancel", r.rect);
            if r.clicked() {
                let _ = app.run("vera.model.cancel", json!({}));
                app.caches.vera_dl_watch = false;
                app.caches.vera_epoch += 1;
            }
        });
        hint(ui, t, "You can close Settings: the download continues, and resumes if interrupted.");
        return;
    }
    hint(
        ui,
        t,
        "Finds photos by what they show (“a dog on a beach”) and pictures that look alike. It isn't part of LightCraft, and everything else works without it.",
    );
    if let Some(e) = model["downloadError"].as_str() {
        let text = crate::i18n::tr_format!("The download didn't work: {e}", e = e);
        let r = ui.add(egui::Label::new(RichText::new(text).font(t.font(11.5)).color(t.caution)).wrap());
        register(ui.ctx(), "vera:downloadError", r.rect);
    }
    let label = crate::i18n::tr_format!("Download the model ({size})", size = gb(bytes));
    let r = ui.add_enabled(model["dir"].is_string(), egui::Button::new(label));
    register(ui.ctx(), "button:veraDownload", r.rect);
    if r.clicked() {
        open_licence(app, model);
    }
    if !model["dir"].is_string() {
        hint(ui, t, "This build has nowhere to keep the model. (The desktop app has.)");
    }
}

/// The model is here: search by meaning on or off for this library, what it takes, and Remove (asked first).
fn installed_rows(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, status: &Value, model: &Value) {
    let mut on = model["state"] != "disabled_by_user";
    if check(ui, "vera.vectorLane", &mut on, "Search by meaning") {
        let _ = app.run("vera.config", json!({"vectorLane": on}));
        app.caches.vera_epoch += 1;
    }
    let bytes = model["bytes"].as_u64().unwrap_or(0);
    let memory = status["model"]["memoryBytes"].as_u64().unwrap_or(0);
    let mut line = crate::i18n::tr_format!("{size} on disk", size = gb(bytes));
    if memory > 0 {
        line.push_str(&crate::i18n::tr_format!(" · {mb} MB in memory while in use", mb = memory / 1_000_000));
    }
    if let Some(dir) = model["dir"].as_str() {
        line.push_str(" · ");
        line.push_str(dir);
    }
    ui.add(egui::Label::new(RichText::new(line).font(t.font(11.5)).color(t.text_dim)).wrap());
    if model["state"] == "error"
        && let Some(why) = status["lanes"]["vector"]["reason"].as_str()
    {
        ui.add(egui::Label::new(RichText::new(why).font(t.font(11.5)).color(t.caution)).wrap());
    }
    let asking = egui::Id::new("vera-remove-asking");
    if ui.ctx().data(|d| d.get_temp::<bool>(asking)).unwrap_or(false) {
        ui.add(
            egui::Label::new(
                RichText::new(crate::i18n::tr_format!(
                    "Remove the model ({size}) and this library's embeddings? Search by meaning stops until you download it again.",
                    size = gb(bytes)
                ))
                .color(t.text_label),
            )
            .wrap(),
        );
        ui.horizontal(|ui| {
            let keep = ui.button(crate::i18n::tr("Keep"));
            register(ui.ctx(), "button:veraRemoveCancel", keep.rect);
            let remove = ui.button(crate::i18n::tr("Remove"));
            register(ui.ctx(), "button:veraRemoveConfirm", remove.rect);
            if keep.clicked() {
                ui.ctx().data_mut(|d| d.remove::<bool>(asking));
            }
            if remove.clicked() {
                ui.ctx().data_mut(|d| d.remove::<bool>(asking));
                match app.run("vera.model.remove", json!({})) {
                    Ok(_) => app.toast(ui.ctx(), crate::i18n::tr("The SigLIP 2 model is removed: search by meaning is off")),
                    Err(e) => app.toast_error(ui.ctx(), e),
                }
                app.caches.vera_epoch += 1;
            }
        });
    } else {
        let r = ui.button(crate::i18n::tr("Remove…"));
        register(ui.ctx(), "button:veraRemove", r.rect);
        if r.clicked() {
            ui.ctx().data_mut(|d| d.insert_temp(asking, true));
        }
    }
}

/// "Download the model": its terms first, in a dialog of their own (it replaces Settings, and goes back there).
pub fn open_licence(app: &mut LightcraftApp, model: &Value) {
    app.ui.dialog = Some(crate::state::Dialog::VeraModel { info: model.clone(), accepted: false, error: None });
}

/// The licence dialog's body: the model, its size and where it goes, the notice, the links, and the accept box.
pub fn licence_dialog(app: &mut LightcraftApp, ui: &mut egui::Ui, t: &Tokens, info: &Value, accepted: &mut bool, error: Option<&str>) {
    let model_id = info["modelId"].as_str().unwrap_or("google/siglip2-base-patch32-256");
    ui.horizontal(|ui| {
        ui.label(RichText::new("SigLIP 2 Base").font(t.semibold(14.0)).color(t.text));
        ui.label(RichText::new(gb(info["bytes"].as_u64().unwrap_or(0))).color(t.text_dim));
    });
    ui.label(RichText::new(model_id).font(t.font(11.5)).color(t.text_dim));
    ui.add(
        egui::Label::new(
            RichText::new(crate::i18n::tr(
                "Search by meaning uses SigLIP 2, Google's image and text model: it finds photos by what they show and pictures that look alike. It isn't part of LightCraft, and everything else works without it.",
            ))
            .color(t.text_label),
        )
        .wrap(),
    );
    if let Some(dir) = info["dir"].as_str() {
        ui.add(egui::Label::new(RichText::new(crate::i18n::tr_format!("Saved in {dir}", dir = dir)).font(t.font(11.5)).color(t.text_dim)).wrap());
    }
    let name = info["licence"]["name"].as_str().unwrap_or("Apache License 2.0");
    ui.label(RichText::new(crate::i18n::tr_format!("{name} · commercial use allowed", name = name)).font(t.semibold(12.5)).color(t.text));
    if let Some(notice) = info["licence"]["notice"].as_str().filter(|s| !s.is_empty()) {
        ui.add(egui::Label::new(RichText::new(notice).color(t.text_label)).wrap());
    }
    ui.horizontal_wrapped(|ui| {
        if let Some(url) = info["licence"]["url"].as_str() {
            let r = ui.link(crate::i18n::tr("Open the model's page")).on_hover_text(url);
            register(ui.ctx(), "link:veraLicence", r.rect);
            if r.clicked()
                && let Err(e) = crate::links::open(app, url)
            {
                app.toast(ui.ctx(), e);
            }
        }
        let r = ui.link(crate::i18n::tr("Read the Apache License 2.0")).on_hover_text(APACHE_URL);
        register(ui.ctx(), "link:veraApache", r.rect);
        if r.clicked()
            && let Err(e) = crate::links::open(app, APACHE_URL)
        {
            app.toast(ui.ctx(), e);
        }
    });
    if let Some(e) = error {
        let text = crate::i18n::tr_format!("The download didn't work: {e}", e = e);
        let r = ui.add(egui::Label::new(RichText::new(text).color(t.caution)).wrap());
        register(ui.ctx(), "veraModel:error", r.rect);
    }
    ui.add_space(4.0);
    check(ui, "veraModel.accept", accepted, "I have read these terms and accept them for my own use");
    ui.label(
        RichText::new(crate::i18n::tr(
            "Nothing is fetched until you accept. The model is kept on this computer only; LightCraft never uploads or shares it.",
        ))
        .font(t.font(11.5))
        .color(t.text_dim),
    );
}

/// The licence dialog's OK (the engine refuses without the acceptance too): start the download, follow it, and go
/// back to Settings ▸ Search, where its progress is.
pub fn download(app: &mut LightcraftApp, accepted: bool) -> Result<Value, String> {
    if !accepted {
        return Err(crate::i18n::tr("Tick the box to accept the model's terms first").to_string());
    }
    let r = app.run("vera.model.download", json!({"acceptLicense": true}))?;
    app.caches.vera_dl_watch = true;
    app.caches.vera_dl_next = 0.0;
    app.caches.vera_epoch += 1;
    app.ui.dialog = Some(crate::state::Dialog::Settings { tab: "search".into() });
    Ok(r)
}

/// Follows the download the user started, at most once a second: says so when the model is in place, or why the
/// download stopped.
pub fn watch(app: &mut LightcraftApp, ctx: &egui::Context, now: f64) {
    if !app.caches.vera_dl_watch || now < app.caches.vera_dl_next {
        return;
    }
    app.caches.vera_dl_next = now + REFRESH_SECS;
    ctx.request_repaint_after(std::time::Duration::from_secs_f64(REFRESH_SECS));
    let Ok(m) = app.session.execute("vera.model.status", &json!({})) else {
        app.caches.vera_dl_watch = false;
        return;
    };
    if m["installed"] == true {
        app.caches.vera_dl_watch = false;
        app.caches.vera_epoch += 1;
        let text = crate::i18n::tr("The SigLIP 2 model is installed: search by meaning is on").to_string();
        app.ui.status = text.clone();
        app.toast(ctx, text);
    } else if m["downloading"].is_null() {
        // stopped: cancelled (from the activity stack or Settings) or failed
        app.caches.vera_dl_watch = false;
        app.caches.vera_epoch += 1;
        if let Some(e) = m["downloadError"].as_str() {
            app.toast_error(ctx, crate::i18n::tr_format!("The download didn't work: {e}", e = e));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vector_lane_speaks_for_itself_and_passes_on_what_it_cannot_say() {
        assert!(vector_reason("needs_model", "engine words", true).contains("one-time download"));
        assert_eq!(vector_reason("needs_model", "engine words", false), "engine words", "no folder: the engine says why");
        assert_eq!(vector_reason("error", "The SigLIP 2 model could not be used: boom", true), "The SigLIP 2 model could not be used: boom");
        assert!(vector_reason("disabled_by_user", "", true).contains("Switched off"));
        assert_eq!(gb(1_511_714_589), "1.5 GB");
        assert_eq!(state_label("live"), "On");
        assert_eq!(state_label("disabled"), "Off");
        assert_eq!(state_label("something new"), "Not available yet");
    }
}
