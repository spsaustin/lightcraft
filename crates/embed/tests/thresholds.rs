//! Gate 2: the embedding thresholds re-measured on the shipped model (`docs/vera-models.md` → Decision).
//!
//! `EMBED_DUPLICATE` and `EMBED_NEAR` in `crates/vera/src/thresholds.rs` were measured on `siglip2-so400m-patch14-384`;
//! cosine scales do not carry across models. This embeds the fixture sets the digest tests use and prints every
//! cosine, for the thresholds to be chosen from. It asserts only what any working model must satisfy.
//!
//! `LIGHTCRAFT_SIGLIP2_DIR=<checkpoint> cargo test -p lightcraft-embed --release --test thresholds -- --nocapture`
//! (skipped without the variable). JPEG re-exports need an encoder this crate's tests do not have; the engine's
//! `vera_gate2_jpeg_re_exports` measures them through the codecs crate.

use std::path::PathBuf;

use lightcraft_embed::{DIM, INPUT, Model};
use lightcraft_scenes::{Scene, demo_library, render_srgb_linear};

fn model_dir() -> Option<PathBuf> {
    match std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR") {
        Some(d) => Some(PathBuf::from(d)),
        None => {
            eprintln!("skipped: set LIGHTCRAFT_SIGLIP2_DIR to a google/siglip2-base-patch32-256 checkpoint");
            None
        }
    }
}

fn srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// A picture: width, height, display-encoded RGB in 0..1.
struct Pic {
    w: usize,
    h: usize,
    data: Vec<[f32; 3]>,
}

/// `scene` at long edge `edge`, keeping its aspect, display-encoded (linear sRGB from the scene, clipped, sRGB curve).
fn render(scene: &Scene, edge: usize) -> Pic {
    let a = scene.aspect();
    let (w, h) = if a >= 1.0 { (edge, ((edge as f32 / a).round() as usize).max(1)) } else { (((edge as f32 * a).round() as usize).max(1), edge) };
    let img = render_srgb_linear(scene.kind, scene.seed, w, h);
    Pic { w, h, data: img.data.iter().map(|p| p.map(srgb)).collect() }
}

/// The central `frac × frac` of a picture.
fn centre_crop(p: &Pic, frac: f32) -> Pic {
    let (cw, ch) = (((p.w as f32 * frac).round() as usize).max(1), ((p.h as f32 * frac).round() as usize).max(1));
    let (x0, y0) = ((p.w - cw) / 2, (p.h - ch) / 2);
    let data = (y0..y0 + ch).flat_map(|y| p.data[y * p.w + x0..y * p.w + x0 + cw].iter().copied()).collect();
    Pic { w: cw, h: ch, data }
}

/// Every value rounded to 8 bits, as any 8-bit file stores it.
fn quantise(p: &Pic) -> Pic {
    Pic { w: p.w, h: p.h, data: p.data.iter().map(|px| px.map(|c| (c * 255.0).round() / 255.0)).collect() }
}

/// For each of `n` output cells along an axis of `len` source pixels: the source pixels it covers and how much of each.
fn coverage(len: usize, n: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = len as f64 / n as f64;
    (0..n)
        .map(|i| {
            let (a, b) = (i as f64 * scale, (i + 1) as f64 * scale);
            let mut cell = Vec::new();
            let mut x = a.floor() as usize;
            while (x as f64) < b && x < len {
                let w = (b.min(x as f64 + 1.0) - a.max(x as f64)) / scale;
                if w > 0.0 {
                    cell.push((x, w as f32));
                }
                x += 1;
            }
            cell
        })
        .collect()
}

/// Resized to exactly [`INPUT`] × [`INPUT`] by area averaging, aspect not kept (as the engine's worker and the SigLIP
/// processor do).
fn squash(p: &Pic) -> Vec<[f32; 3]> {
    let (cols, rows) = (coverage(p.w, INPUT), coverage(p.h, INPUT));
    let mut mid = vec![[0f32; 3]; p.h * INPUT];
    for y in 0..p.h {
        for (ox, cell) in cols.iter().enumerate() {
            for &(x, wt) in cell {
                for c in 0..3 {
                    mid[y * INPUT + ox][c] += p.data[y * p.w + x][c] * wt;
                }
            }
        }
    }
    let mut out = vec![[0f32; 3]; INPUT * INPUT];
    for (oy, cell) in rows.iter().enumerate() {
        for &(y, wt) in cell {
            for ox in 0..INPUT {
                for c in 0..3 {
                    out[oy * INPUT + ox][c] += mid[y * INPUT + ox][c] * wt;
                }
            }
        }
    }
    out
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn embed(m: &Model, pics: &[Vec<[f32; 3]>]) -> Vec<Vec<f32>> {
    let refs: Vec<&[[f32; 3]]> = pics.iter().map(Vec::as_slice).collect();
    let out = m.embed_images(&refs).unwrap();
    for e in &out {
        assert_eq!(e.len(), DIM);
        assert!((e.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs() < 1e-4);
    }
    out
}

#[test]
fn gate2_embedding_thresholds() {
    let Some(dir) = model_dir() else { return };
    let m = Model::load(&dir, lightcraft_embed::best_device()).unwrap();
    let lib = demo_library();
    let scene = &lib[0];
    println!("Gate 2 on {} ({}), fixture scene {:?} ({:.2}:1)", lightcraft_embed::MODEL_ID, dir.display(), scene.name, scene.aspect());

    // the same picture at four sizes, each squashed to the model input
    let edges = [2048usize, 1024, 512, 256];
    let renders: Vec<Pic> = edges.iter().map(|e| render(scene, *e)).collect();
    let mut inputs: Vec<Vec<[f32; 3]>> = renders.iter().map(squash).collect();
    // an 8-bit copy (what any 8-bit export stores) and a 70 % centre crop, both from the 1024 render
    inputs.push(squash(&quantise(&renders[1])));
    inputs.push(squash(&centre_crop(&renders[1], 0.7)));
    let e = embed(&m, &inputs);
    let (mut same_min, mut same_max) = (f32::MAX, f32::MIN);
    println!("same picture, resized:");
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let c = cos(&e[i], &e[j]);
            (same_min, same_max) = (same_min.min(c), same_max.max(c));
            println!("  {} vs {}: {c:.4}", edges[i], edges[j]);
        }
    }
    let q8 = cos(&e[1], &e[4]);
    println!("  1024 vs its 8-bit copy: {q8:.4}");
    println!("  resize set: {same_min:.4} ..= {same_max:.4}");
    println!("JPEG re-exports: skipped here (no JPEG encoder among this crate's test dependencies; see the engine's vera_gate2_jpeg_re_exports)");
    let crop = cos(&e[1], &e[5]);
    println!("70 % centre crop vs the 1024 frame: {crop:.4}");

    // eight unrelated scenes (the digest tests' set: the demo library's first eight), at 512 on the long edge
    let picked: Vec<&Scene> = lib.iter().take(8).collect();
    let unrelated: Vec<Vec<[f32; 3]>> = picked.iter().map(|s| squash(&render(s, 512))).collect();
    let u = embed(&m, &unrelated);
    let (mut lo, mut hi, mut lo_pair, mut hi_pair) = (f32::MAX, f32::MIN, String::new(), String::new());
    println!("unrelated pairs:");
    for i in 0..u.len() {
        for j in i + 1..u.len() {
            let c = cos(&u[i], &u[j]);
            let pair = format!("{} vs {}", picked[i].name, picked[j].name);
            println!("  {pair}: {c:.4}");
            if c < lo {
                (lo, lo_pair) = (c, pair.clone());
            }
            if c > hi {
                (hi, hi_pair) = (c, pair);
            }
        }
    }
    println!("  unrelated: min {lo:.4} ({lo_pair}), max {hi:.4} ({hi_pair})");
    println!(
        "summary: same picture {same_min:.4}..={same_max:.4} (8-bit {q8:.4}), crop {crop:.4}, unrelated {lo:.4}..={hi:.4}; in force: EMBED_DUPLICATE {}, EMBED_NEAR {} (so400m)",
        0.82, 0.78
    );

    // what any working model satisfies
    assert!(same_min > 0.9, "a resize is the same picture: {same_min}");
    assert!(same_min > hi, "every resize pair is closer than every unrelated pair: {same_min} vs {hi}");
    assert!(crop < same_min, "a crop is further than a resize: {crop} vs {same_min}");
}
