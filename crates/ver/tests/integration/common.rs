//! Shared helpers for the integration tests.
#![allow(dead_code)]

use lightcraft_ver::cpnp::{self, Format};
use lightcraft_ver::hashing;
use lightcraft_ver::record::{Builder, Level, Record, Schema};
use lightcraft_ver::segments;
use serde_json::json;
use std::path::PathBuf;

pub fn standard() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("standard")
}

pub fn corpus_file(name: &str) -> Vec<u8> {
    std::fs::read(standard().join("decoder-corpus/1/corpus").join(name)).expect("corpus file")
}

/// A visual space descriptor valid under both schemas (no `reference`: VER404 is a warning).
pub fn space(dim: usize, dtype: &str) -> serde_json::Value {
    json!({
        "space_id": "org.lightcraft.space.test-visual/1.0",
        "kind": "visual",
        "model": { "family": "test", "architecture": "none", "checkpoint": "test-1", "weights_sha256": "0".repeat(64) },
        "dim": dim,
        "dtype": dtype,
        "normalization": "l2",
        "metric": "cosine",
        "preprocessing": { "resize": "none", "interpolation": "none", "antialias": false, "crop": "none", "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "channel_order": "rgb" }
    })
}

/// An L3 record over `bytes` built with every builder setter.
pub fn l3_record(bytes: &[u8], schema: Schema) -> Result<Record, String> {
    let (buf, id) = hashing::identity(bytes).map_err(|e| e.to_string())?;
    let (avail, raw) = segments::segments(bytes, buf.format);
    let vector: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
    Builder::new(schema, "lightcraft-ver-test", "0.0.0")
        .identity(&id, buf.format, buf.animated, buf.alpha_present, buf.hdr_present)
        .source_bytes_ref("https://example.org/assets/test", &id.content_hash, bytes.len() as u64)
        .availability(&avail)
        .raw_segments(&raw)
        .normalized(
            json!({ "@context": { "dc": "http://purl.org/dc/elements/1.1/" }, "dc:description": "a test picture" }),
            vec![("dc:description".into(), "unsigned".into())],
        )
        .space(space(8, "fp16"))
        .embedding("canonical_visual", "org.lightcraft.space.test-visual/1.0", 8, "fp16", &vector, "2026-10-11T10:00:03Z", "lightcraft-ver-test")
        .acquisition("user_upload", "2026-10-11T10:00:00Z", json!({ "actor": "tester" }))
        .chain_event("decode", "lightcraft", "lightcraft-ver cpnp-1", "2026-10-11T10:00:01Z", json!({}))
        .chain_event("normalize", "lightcraft", "lightcraft-ver cpnp-1", "2026-10-11T10:00:02Z", json!({}))
        .chain_event("embed", "lightcraft", "test", "2026-10-11T10:00:03Z", json!({}))
        .c2pa_status("missing")
        .level(Level::L3)
        .record_id("0b6f6e3a-8e0c-5a43-9a51-1c2d3e4f5a6b")
        .created_at("2026-10-11T10:00:04Z")
        .build()
}

pub fn format_of(bytes: &[u8]) -> Format {
    cpnp::is_cpnp_format(bytes).expect("a CPNP format")
}
