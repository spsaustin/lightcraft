//! The published decoder corpus and reference-set pixel vectors, and the evaluation document.
//!
//! [`run_decoder_corpus`] follows `standard/decoder-corpus/1/README.md` → "Evaluating an
//! implementation": check each Asset's `content_sha256` first (a mismatch means it is not the
//! Asset the corpus describes, so the entry fails without being decoded), canonicalise, compare
//! dimensions and `pixel_hash`. [`render_evaluation`] writes the result in the shape that
//! README's "Publishing an evaluation" asks for.

use crate::cpnp;
use crate::hashing;
use serde_json::Value;
use std::path::Path;

/// How many of `ver-decoder-corpus/1`'s 21 entries this implementation reproduces: the count the
/// corpus test pins and `cargo xtask ver corpus` fails below, so a regression is caught.
pub const PINNED_DECODER_CORPUS: usize = 21;

/// How many of the ten reference-set images reproduce their `pixel_sha256`.
pub const PINNED_REFERENCE_SET_PIXELS: usize = 10;

/// The vendored `decoder-corpus/1` directory, relative to the workspace root.
pub const DECODER_CORPUS_DIR: &str = "crates/ver/standard/decoder-corpus/1";

/// The vendored `reference-sets` directory, relative to the workspace root.
pub const REFERENCE_SETS_DIR: &str = "crates/ver/standard/reference-sets";

/// One corpus entry's outcome.
#[derive(Clone, Debug)]
pub struct CorpusEntry {
    pub path: String,
    pub expected: String,
    pub got: Option<String>,
    pub ok: bool,
    pub note: String,
}

/// A whole run.
#[derive(Clone, Debug)]
pub struct CorpusReport {
    pub corpus: String,
    pub profile: String,
    pub total: usize,
    pub reproduced: usize,
    pub entries: Vec<CorpusEntry>,
    pub toolchain: Vec<(String, String)>,
}

/// The decode toolchain of this build, in the shape of `PROFILE.md` §2.
pub fn toolchain() -> Vec<(String, String)> {
    let mut v = vec![
        (
            "lightcraft-ver".to_string(),
            format!("{} (CPNP-1 driver; JPEG decoded by its own libjpeg-turbo-equivalent islow/fancy/jdcolor path)", env!("CARGO_PKG_VERSION")),
        ),
        ("lightcraft-codecs".to_string(), format!("{} (format sniffing only)", env!("CARGO_PKG_VERSION"))),
    ];
    let role = |name: &str| match name {
        "png" => " (PNG decode)",
        "image-webp" => " (WebP decode)",
        "tiff" => " (TIFF decode, LZW)",
        "gif" => " (GIF frame 0 LZW decode)",
        "image" => " (BMP decode)",
        "moxcms" => " (ICC → sRGB, relative colorimetric)",
        "miniz_oxide" => " (iCCP / zTXt inflate)",
        "zune-jpeg" | "jpeg-decoder" => " (in the workspace, not on the CPNP-1 path: evaluated, see docs/ver-cpnp.md)",
        _ => "",
    };
    for item in env!("VER_TOOLCHAIN_CRATES").split(';') {
        if let Some((name, version)) = item.split_once('=') {
            v.push((name.to_string(), format!("{version}{}", role(name))));
        }
    }
    v.push(("rustc".to_string(), env!("VER_RUSTC_VERSION").to_string()));
    v
}

fn read_manifest(dir: &Path) -> Result<Value, String> {
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}

/// Evaluate `dir` (a `decoder-corpus/1` directory).
pub fn run_decoder_corpus(dir: &Path) -> Result<CorpusReport, String> {
    let manifest = read_manifest(dir)?;
    let files = manifest.get("files").and_then(Value::as_array).ok_or("manifest.json has no files array")?;
    let mut entries = Vec::new();
    for f in files {
        let path = s(f, "path");
        let expected = s(f, "pixel_hash");
        let want_w = f.get("width").and_then(Value::as_u64);
        let want_h = f.get("height").and_then(Value::as_u64);
        let mut entry = CorpusEntry { path: path.clone(), expected: expected.clone(), got: None, ok: false, note: String::new() };
        let bytes = match std::fs::read(dir.join(&path)) {
            Ok(b) => b,
            Err(e) => {
                entry.note = format!("cannot read: {e}");
                entries.push(entry);
                continue;
            }
        };
        if hashing::content_hash(&bytes) != s(f, "content_sha256") {
            entry.note = "content_sha256 mismatch: not the Asset this corpus describes".into();
            entries.push(entry);
            continue;
        }
        match cpnp::canonicalize(&bytes) {
            Ok(buf) => {
                let got = hashing::pixel_hash(&buf);
                let dims_ok = want_w == Some(u64::from(buf.width)) && want_h == Some(u64::from(buf.height));
                entry.ok = dims_ok && got == expected;
                entry.note = if dims_ok {
                    format!("{}×{}", buf.width, buf.height)
                } else {
                    format!("dimensions {}×{}, expected {}×{}", buf.width, buf.height, want_w.unwrap_or(0), want_h.unwrap_or(0))
                };
                entry.got = Some(got);
            }
            Err(e) => entry.note = format!("no Canonical Buffer: {e}"),
        }
        entries.push(entry);
    }
    let reproduced = entries.iter().filter(|e| e.ok).count();
    Ok(CorpusReport {
        corpus: s(&manifest, "corpus"),
        profile: s(&manifest, "profile"),
        total: entries.len(),
        reproduced,
        entries,
        toolchain: toolchain(),
    })
}

/// Check the reference sets' PNG pixel vectors under `dir` (a `reference-sets` directory): every
/// image any of the three manifests lists, deduplicated by path, against its `content_sha256`
/// and `pixel_sha256` (the reference sets' name for `pixel_hash`).
pub fn run_reference_set_pixels(dir: &Path) -> Result<CorpusReport, String> {
    let mut seen: Vec<(String, String, String)> = Vec::new();
    let mut set_ids = Vec::new();
    let mut subdirs: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    subdirs.sort();
    for sub in subdirs {
        let manifest = read_manifest(&sub)?;
        if let Some(id) = manifest.get("space_id").and_then(Value::as_str) {
            set_ids.push(id.to_string());
        }
        let items = manifest.get("items").and_then(Value::as_array).ok_or_else(|| format!("{}: no items array", sub.display()))?;
        for img in items {
            let path = format!("images/{}", s(img, "name"));
            if !seen.iter().any(|(p, _, _)| *p == path) {
                seen.push((path, s(img, "content_sha256"), s(img, "pixel_sha256")));
            }
        }
    }
    let mut entries = Vec::new();
    for (path, content, pixel) in seen {
        let mut entry = CorpusEntry { path: path.clone(), expected: pixel.clone(), got: None, ok: false, note: String::new() };
        match std::fs::read(dir.join(&path)) {
            Err(e) => entry.note = format!("cannot read: {e}"),
            Ok(bytes) if hashing::content_hash(&bytes) != content => entry.note = "content_sha256 mismatch".into(),
            Ok(bytes) => match cpnp::canonicalize(&bytes) {
                Ok(buf) => {
                    let got = hashing::pixel_hash(&buf);
                    entry.ok = got == pixel;
                    entry.note = format!("{}×{}", buf.width, buf.height);
                    entry.got = Some(got);
                }
                Err(e) => entry.note = format!("no Canonical Buffer: {e}"),
            },
        }
        entries.push(entry);
    }
    let reproduced = entries.iter().filter(|e| e.ok).count();
    Ok(CorpusReport {
        corpus: format!("reference-sets ({})", set_ids.join(", ")),
        profile: "CPNP-1 pixel_sha256".into(),
        total: entries.len(),
        reproduced,
        entries,
        toolchain: toolchain(),
    })
}

/// The published evaluation, as Markdown: profile, corpus, our toolchain (PROFILE.md §2's
/// shape), the count reproduced, and every failure's path, expected and got.
pub fn render_evaluation(report: &CorpusReport) -> String {
    let mut md = String::new();
    md.push_str(&format!("| | |\n| --- | --- |\n| Profile identifier | `{}` |\n| Corpus identifier | `{}` |\n", report.profile, report.corpus));
    md.push_str(&format!("| Reproduced | **{} of {}** |\n", report.reproduced, report.total));
    let verdict = if report.total > 0 && report.reproduced == report.total {
        format!("conformant for `{}`", report.profile)
    } else {
        format!("**not** conformant for `{}`: identity claims stay PROVISIONAL (PROFILE.md §4)", report.profile)
    };
    md.push_str(&format!("| Verdict | {verdict} |\n\n"));
    md.push_str("Toolchain (the shape of PROFILE.md §2):\n\n| Component | Version (role) |\n| --- | --- |\n");
    for (name, version) in &report.toolchain {
        md.push_str(&format!("| {name} | {version} |\n"));
    }
    md.push_str("\n| Path | Result | Note |\n| --- | --- | --- |\n");
    for e in &report.entries {
        md.push_str(&format!("| `{}` | {} | {} |\n", e.path, if e.ok { "reproduced" } else { "**FAIL**" }, e.note));
    }
    let failures: Vec<&CorpusEntry> = report.entries.iter().filter(|e| !e.ok).collect();
    if failures.is_empty() {
        md.push_str("\nNo failures.\n");
    } else {
        md.push_str("\nFailures:\n\n| Path | Expected `pixel_hash` | Got |\n| --- | --- | --- |\n");
        for e in failures {
            md.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                e.path,
                e.expected,
                e.got.as_deref().map(|g| format!("`{g}`")).unwrap_or_else(|| e.note.clone())
            ));
        }
    }
    md
}
