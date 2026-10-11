//! LightCraft's embedding space, as a VER space descriptor, and the pixel path that feeds it.
//!
//! The space is `<authority>.space.siglip2-base-p32-256/1.0`: SigLIP 2 Base patch32-256 (the checkpoint the vector
//! lane downloads), fed from the CPNP-1 Canonical Buffer resized to 256 × 256 by a bilinear (triangle) filter whose
//! support widens with the downscale (Pillow's `BILINEAR`, the filter the Hugging Face SigLIP processor uses), with no
//! crop, rescaled by 1/255 and normalized with mean and standard deviation 0.5 per channel, stored as l2-normalised
//! fp16. Its descriptor carries **no `reference` block**: no tolerance has been measured for it, so it makes no
//! conformance claim (VER 1.1 draft §8.2, errata E13) and its vectors are comparable only within this producer.
//!
//! `weights_sha256` is SHA-256 over `model.safetensors` (errata E29: producer-scoped, comparable only within one
//! producer); it is computed once per weights file and remembered beside it.

// the web build embeds nothing (no model): the descriptor and the pixel path are the desktop's
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use std::path::Path;

use serde_json::{Value, json};

/// The checkpoint LightCraft downloads (`lightcraft_embed::fetch`).
pub(crate) const CHECKPOINT_P32: &str = "google/siglip2-base-patch32-256";
/// The space id's local part (after the authority).
pub(crate) const SPACE_LOCAL_P32: &str = "space.siglip2-base-p32-256/1.0";
/// When LightCraft's descriptor was published: fixed, because a descriptor is immutable once published.
pub(crate) const PUBLISHED_AT: &str = "2026-10-11T00:00:00Z";
/// Embedding dimensions and storage type of the space.
pub(crate) const DIM: usize = 768;
pub(crate) const DTYPE: &str = "fp16";
/// The model input's side.
pub(crate) const INPUT: usize = 256;

/// The space id under `authority` (`org.verstandard` → `org.verstandard.space.siglip2-base-p32-256/1.0`).
pub(crate) fn space_id(authority: &str) -> String {
    format!("{authority}.{SPACE_LOCAL_P32}")
}

/// The domain an authority names (`org.verstandard` → `verstandard.org`).
pub(crate) fn domain(authority: &str) -> String {
    authority.split('.').rev().collect::<Vec<_>>().join(".")
}

/// Whether `a` can be a space-id authority under both schemas: a reverse-DNS name of at least two labels, lowercase
/// letters, digits and inner hyphens, the first label without hyphens (the 1.0.1 pattern), and short enough to be a
/// domain.
pub(crate) fn valid_authority(a: &str) -> bool {
    let labels: Vec<&str> = a.split('.').collect();
    let label_ok = |l: &str| {
        !l.is_empty()
            && l.len() <= 63
            && !l.starts_with('-')
            && !l.ends_with('-')
            && l.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    };
    a.len() <= 200 && labels.len() >= 2 && labels.iter().all(|l| label_ok(l)) && labels.first().is_some_and(|l| !l.contains('-'))
}

/// The visual preprocessing members, named as the reference producer (`verd.space_cache.preprocessing_from_processor`)
/// names them for its SigLIP 2 processor, plus `antialias` and `crop` so the descriptor states every member errata
/// E13 lists for a visual space.
pub(crate) fn preprocessing() -> Value {
    json!({
        "resize": {"height": INPUT, "width": INPUT},
        "do_resize": true,
        "interpolation": "bilinear",
        "antialias": true,
        "crop": "none",
        "do_center_crop": false,
        "do_rescale": true,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": true,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "channel_order": "rgb",
    })
}

/// LightCraft's space descriptor under `authority`, pinned to the weights whose SHA-256 is `weights_sha256`.
pub(crate) fn descriptor(authority: &str, weights_sha256: &str) -> Value {
    json!({
        "space_id": space_id(authority),
        "kind": "visual",
        "model": {
            "family": "siglip2",
            "architecture": "ViT-B/32",
            "checkpoint": CHECKPOINT_P32,
            "weights_sha256": weights_sha256,
        },
        "dim": DIM,
        "dtype": DTYPE,
        "normalization": "l2",
        "metric": "cosine",
        "preprocessing": preprocessing(),
        "provenance": {"publisher": domain(authority), "published_at": PUBLISHED_AT, "license": "Apache-2.0"},
    })
}

/// SHA-256 of the weights file at `path`, remembered in `<path>.ver-sha256` against the file's size and modification
/// time (hashing 1.5 GB takes seconds; it is done once per file). An error when the file cannot be read.
pub(crate) fn weights_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stamp =
        format!("{} {}", meta.len(), meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs()));
    let cache = path.with_extension("safetensors.ver-sha256");
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Some((s, h)) = text.trim().rsplit_once(' ')
        && s == stamp
        && h.len() == 64
        && h.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Ok(h.to_ascii_lowercase());
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(buf.get(..n).unwrap_or(&[]));
    }
    let hex = super::util::hex(&h.finalize());
    if let Err(e) = std::fs::write(&cache, format!("{stamp} {hex}\n")) {
        log::debug!("ver: could not remember the weights digest in {}: {e}", cache.display());
    }
    Ok(hex)
}

/// Pillow's fixed-point precision for 8-bit resampling.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// The coefficients of a bilinear (triangle) resample of an axis of `in_size` samples to `out_size`, as Pillow
/// computes them (`precompute_coeffs` + `normalize_coeffs_8bpc`): for each output sample, the first input sample and
/// the fixed-point weights.
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size.max(1) as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor() as usize).min(in_size);
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|x| {
                    let t = ((x as f64 - center + 0.5) * ss).abs();
                    if t < 1.0 { 1.0 - t } else { 0.0 }
                })
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                for v in &mut w {
                    *v /= total;
                }
            }
            let one = f64::from(1u32 << PRECISION_BITS);
            let fixed = w.iter().map(|v| if *v < 0.0 { (-0.5 + v * one) as i32 } else { (0.5 + v * one) as i32 }).collect();
            (xmin, fixed)
        })
        .collect()
}

fn clip8(acc: i64) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resize interleaved RGB8 `rgb` (`w` × `h`) to `side` × `side` as Pillow's `Image.resize(.., BILINEAR)` does: a
/// horizontal pass to 8-bit, then a vertical one, triangle weights in fixed point. `None` for an empty or short buffer.
pub(crate) fn resize_bilinear(rgb: &[u8], w: usize, h: usize, side: usize) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || side == 0 || rgb.len() < w.checked_mul(h)?.checked_mul(3)? {
        return None;
    }
    let half = 1i64 << (PRECISION_BITS - 1);
    let cx = coeffs(w, side);
    let mut mid = vec![0u8; h.checked_mul(side)?.checked_mul(3)?];
    for y in 0..h {
        let row = rgb.get(y * w * 3..(y + 1) * w * 3)?;
        for (xx, (xmin, k)) in cx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = half;
                for (i, kv) in k.iter().enumerate() {
                    acc += i64::from(*row.get((xmin + i) * 3 + c)?) * i64::from(*kv);
                }
                *mid.get_mut((y * side + xx) * 3 + c)? = clip8(acc);
            }
        }
    }
    let cy = coeffs(h, side);
    let mut out = vec![0u8; side * side * 3];
    for (yy, (ymin, k)) in cy.iter().enumerate() {
        for x in 0..side {
            for c in 0..3 {
                let mut acc = half;
                for (i, kv) in k.iter().enumerate() {
                    acc += i64::from(*mid.get(((ymin + i) * side + x) * 3 + c)?) * i64::from(*kv);
                }
                *out.get_mut((yy * side + x) * 3 + c)? = clip8(acc);
            }
        }
    }
    Some(out)
}

/// The model input for a Canonical Buffer: resized ([`resize_bilinear`]) and rescaled to 0..1 (the model applies the
/// mean and standard deviation itself).
pub(crate) fn model_input(rgb: &[u8], w: usize, h: usize) -> Option<Vec<[f32; 3]>> {
    let small = resize_bilinear(rgb, w, h, INPUT)?;
    Some(small.chunks_exact(3).map(|p| [f32::from(p[0]) / 255.0, f32::from(p[1]) / 255.0, f32::from(p[2]) / 255.0]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorities_and_ids() {
        assert!(valid_authority("org.verstandard"));
        assert!(valid_authority("com.example-studio.photos"));
        assert!(!valid_authority("verstandard"));
        assert!(!valid_authority("Org.verstandard"));
        assert!(!valid_authority("my-org.example"), "the 1.0.1 pattern has no hyphen in the first label");
        assert!(!valid_authority("org..x"));
        assert!(!valid_authority("org.x-"));
        assert_eq!(space_id("org.verstandard"), "org.verstandard.space.siglip2-base-p32-256/1.0");
        assert_eq!(domain("org.verstandard"), "verstandard.org");
        let d = descriptor("org.verstandard", &"a".repeat(64));
        assert!(d.get("reference").is_none(), "no conformance claim until a tolerance is measured");
        for k in ["resize", "interpolation", "antialias", "crop", "mean", "std", "channel_order"] {
            assert!(d["preprocessing"].get(k).is_some(), "{k}");
        }
    }

    #[test]
    fn bilinear_resize_is_pillow_shaped() {
        // a flat picture stays flat at any scale
        let flat = vec![100u8; 300 * 200 * 3];
        let out = resize_bilinear(&flat, 300, 200, 16).unwrap();
        assert!(out.iter().all(|v| *v == 100));
        // identity size is the identity
        let ramp: Vec<u8> = (0..8 * 8 * 3).map(|i| (i * 3 % 256) as u8).collect();
        assert_eq!(resize_bilinear(&ramp, 8, 8, 8).unwrap(), ramp);
        // halving a 2-pixel period averages it (antialiased): black/white columns → mid grey
        let stripes: Vec<u8> = (0..64 * 64).flat_map(|i| if i % 2 == 0 { [0u8; 3] } else { [255u8; 3] }).collect();
        let half = resize_bilinear(&stripes, 64, 64, 32).unwrap();
        // away from the edges, where the window is cut short and renormalised, as Pillow's is
        for y in 1..31 {
            for x in 1..31 {
                let v = half[(y * 32 + x) * 3];
                assert!((v as i32 - 128).abs() <= 1, "({x}, {y}): {v}");
            }
        }
        // weights sum to one in fixed point
        for (w, s) in [(1000, 256), (256, 256), (100, 256), (3, 256)] {
            for (_, k) in coeffs(w, s) {
                let sum: i32 = k.iter().sum();
                assert!((sum - (1 << PRECISION_BITS)).abs() <= k.len() as i32, "{w}->{s}: {sum}");
            }
        }
        assert!(resize_bilinear(&[], 0, 0, 4).is_none());
        assert!(resize_bilinear(&[1, 2, 3], 2, 2, 4).is_none());
        assert_eq!(model_input(&flat, 300, 200).unwrap().len(), INPUT * INPUT);
    }
}
