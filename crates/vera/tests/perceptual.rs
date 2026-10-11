//! `lightcraft_vera::digest` against the reference implementations errata E22 names: Meta's PDQ
//! (through ThreatExchange's `pdq_hasher`) and `imagehash.phash`. The vectors and the script that
//! generated them (`gen_perceptual_vectors.py`) are beside this file. Here: the ten reference-set
//! images (plain RGB PNGs, read as they are) and the synthetic patterns the script also hashes;
//! the 21 decoder-corpus Canonical Buffers need CPNP-1 and are checked by `lightcraft-ver`.

use lightcraft_vera::digest::{self, Planes};
use std::path::PathBuf;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The script's `synthetic(w, h)`.
fn synthetic_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 255) / (w - 1).max(1)) as u8);
            v.push((((x ^ y) * 3) & 255) as u8);
            v.push((((x * x + y * 7) >> 3) & 255) as u8);
        }
    }
    v
}

#[test]
fn reference_vectors_reproduce_exactly() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tests/perceptual-vectors.json")).expect("vectors")).expect("json");
    let mut checked = 0;
    for v in doc["vectors"].as_array().expect("vectors") {
        let path = v["path"].as_str().expect("path");
        let (w, h) = (v["width"].as_u64().expect("width") as usize, v["height"].as_u64().expect("height") as usize);
        let rgb = if path.starts_with("synthetic:") {
            synthetic_rgb(w, h)
        } else if path.starts_with("reference-sets/") {
            let img = image::open(dir.join("../ver/standard").join(path)).expect("png").to_rgb8();
            assert_eq!((img.width() as usize, img.height() as usize), (w, h), "{path}");
            img.into_raw()
        } else {
            continue;
        };
        let planes = Planes::from_rgb8(w, h, &rgb).expect("planes");
        assert_eq!(hex(&digest::phash_resample(&planes).expect("resample")), v["l32"].as_str().expect("l32"), "{path}: PIL Lanczos 32×32");
        let d = digest::digest_planes(&planes).expect("digest");
        assert_eq!(format!("{:016x}", d.phash), v["phash"].as_str().expect("phash"), "{path}: phash");
        assert_eq!(hex(&d.pdq), v["pdq"].as_str().expect("pdq"), "{path}: pdq");
        assert_eq!(u64::from(d.pdq_quality), v["pdq_quality"].as_u64().expect("quality"), "{path}: pdq quality");
        assert_eq!(digest::digest_rgb8(w, h, &rgb), Some(d));
        checked += 1;
    }
    assert_eq!(checked, 15, "10 reference-set images and 5 synthetic patterns");
}
