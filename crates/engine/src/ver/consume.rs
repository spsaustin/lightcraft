//! Reading a VER Record from anywhere (`ver.import`): validated under its own schema, then matched against the
//! library.
//!
//! A Record with an Error-severity issue is **not accepted**: its issues are reported and nothing is matched. An
//! accepted Record is matched by:
//! - `content`: its `content_hash` equals a library file's SHA-256 (the same bytes; dedupe belongs to this tier, E27);
//! - `pixel_hash`: the same rendered pixels under CPNP-1 (a *grouping*: distinct files that render identically share
//!   it; errata E27 forbids reading it as an identity);
//! - `pdq` / `phash`: its perceptual digests within the same-picture thresholds of `lightcraft_vera::thresholds`
//!   (`score` is the Hamming distance; from an unfamiliar producer a within-producer clustering signal only, E22);
//! - `vector`: only when its `canonical_visual` embedding is in LightCraft's own space, with the same weights pin
//!   (E29: comparable only within one producer; the space has no reference set, so this is never a cross-producer
//!   claim, VER 1.1 §8.2). `score` is the cosine.

use lightcraft_ver::record::Record;
use lightcraft_ver::validate::Issue;
use lightcraft_vera::thresholds;
use serde_json::{Value, json};

use super::VerError;
use crate::Session;
use crate::media::content_key;

/// Largest Record text read (the 1.1 draft's §7(a) limit).
pub(crate) const MAX_RECORD_BYTES: usize = 10 << 20;

/// One match.
#[derive(Clone, Debug)]
pub struct Match {
    pub id: u64,
    /// `content` | `pixel_hash` | `pdq` | `phash` | `vector`.
    pub by: &'static str,
    pub score: f64,
}

/// What reading a Record found.
#[derive(Clone, Debug)]
pub struct Consumed {
    /// `accepted` or `rejected`.
    pub status: &'static str,
    pub issues: Vec<Issue>,
    pub matches: Vec<Match>,
    pub summary: Value,
    /// Library pictures not fingerprinted yet (they cannot match until they are).
    pub pending: usize,
}

impl Consumed {
    pub fn to_json(&self) -> Value {
        json!({
            "status": self.status,
            "accepted": self.status == "accepted",
            "issues": self.issues,
            "matches": self.matches.iter().map(|m| json!({"id": m.id, "by": m.by, "score": m.score})).collect::<Vec<_>>(),
            "recordSummary": self.summary,
            "pendingDigests": self.pending,
        })
    }
}

fn rank(by: &str) -> u8 {
    match by {
        "content" => 0,
        "pixel_hash" => 1,
        "pdq" => 2,
        "phash" => 3,
        _ => 4,
    }
}

/// A short account of a Record for people and agents.
fn summary(r: &Record) -> Value {
    let d = &r.0;
    let spaces: Vec<Value> = r.spaces().iter().filter_map(|s| s.get("space_id").cloned()).collect();
    json!({
        "recordId": d.get("record_id"),
        "verVersion": d.get("ver_version"),
        "level": d.get("conformance_level"),
        "generator": d.get("generator"),
        "contentHash": d.pointer("/identity/content_hash/value"),
        "pixelHash": d.pointer("/identity/pixel_hash/value"),
        "width": d.pointer("/identity/pixel_hash/width"),
        "height": d.pointer("/identity/pixel_hash/height"),
        "format": d.pointer("/image/format"),
        "spaces": spaces,
        "parents": d.pointer("/lineage/parents").and_then(Value::as_array).map_or(0, Vec::len),
        "signed": d.pointer("/provenance/signature").is_some(),
    })
}

/// Read `text` as a Record, validate it and match it against the library.
pub(crate) fn consume(s: &mut Session, text: &str) -> Result<Consumed, VerError> {
    if text.len() > MAX_RECORD_BYTES {
        return Err(VerError::Invalid(format!("a Record is at most {MAX_RECORD_BYTES} bytes (VER 1.1 draft §7)")));
    }
    let record = Record::parse(text).map_err(|e| VerError::Invalid(format!("not a VER Record: {e}")))?;
    let schema =
        record.schema().ok_or_else(|| VerError::Invalid("not a VER Record LightCraft reads: ver_version must be \"1.0\" or \"1.1\"".into()))?;
    // schema, profile, and the core's own consistency rules (LCVER-E2: the pixel hash and the image state one size)
    let issues = lightcraft_ver::validate::all(&record.0, schema);
    let summary = summary(&record);
    if !lightcraft_ver::validate::is_conformant(&issues) {
        return Ok(Consumed { status: "rejected", issues, matches: Vec::new(), summary, pending: 0 });
    }
    crate::vera::ensure_open(s);
    let id = record.identity();
    let parse32 = |h: Option<&str>| h.and_then(super::util::unhex).and_then(|v| <[u8; 32]>::try_from(v).ok());
    let content = parse32(id.as_ref().map(|i| i.content_hash.as_str()));
    let pixel = parse32(id.as_ref().map(|i| i.pixel_hash.as_str()));
    let pdq = parse32(id.as_ref().and_then(|i| i.pdq.as_deref()));
    let phash = id.as_ref().and_then(|i| i.phash.as_deref()).filter(|p| p.len() == 16).and_then(|p| u64::from_str_radix(p, 16).ok());
    // the record's vector, when it is in this producer's space with the same weights
    let ours = super::embedding_source(s).space().ok();
    let vector: Option<Vec<f32>> = ours.as_ref().and_then(|sp| {
        let same_space = record.spaces().iter().any(|d| {
            d.get("space_id").and_then(Value::as_str) == Some(sp.space_id.as_str())
                && d.pointer("/model/weights_sha256") == sp.descriptor.pointer("/model/weights_sha256")
        });
        if !same_space {
            return None;
        }
        record.embeddings().into_iter().find(|(role, space, ..)| role == "canonical_visual" && *space == sp.space_id).and_then(|(.., v)| v)
    });
    let lane_space = super::space::space_id(&s.ver.prefs.authority());
    let mut matches = Vec::new();
    let mut pending = 0usize;
    let photos: Vec<(u64, String)> = s.catalog.photos().filter(|p| crate::vera::eligible(p)).map(|p| (p.id.0, content_key(p))).collect();
    for (photo, key) in photos {
        let Some(fp) = s.vera.digests.fingerprint(&key).copied() else {
            pending += 1;
            continue;
        };
        if content.is_some() && fp.content == content {
            matches.push(Match { id: photo, by: "content", score: 1.0 });
        }
        if pixel.is_some() && fp.pixel == pixel {
            matches.push(Match { id: photo, by: "pixel_hash", score: 1.0 });
        }
        let d_pdq = pdq.map(|p| lightcraft_vera::digest::hamming256(&p, &fp.digest.pdq));
        let d_phash = phash.map(|p| lightcraft_vera::digest::hamming64(p, fp.digest.phash));
        let pdq_trusted = fp.digest.pdq_quality >= thresholds::PDQ_MIN_QUALITY;
        match (d_pdq, d_phash) {
            (Some(d), _) if pdq_trusted && d <= thresholds::PDQ_TIGHT => matches.push(Match { id: photo, by: "pdq", score: f64::from(d) }),
            (_, Some(d)) if d <= thresholds::PHASH_TIGHT => matches.push(Match { id: photo, by: "phash", score: f64::from(d) }),
            _ => {}
        }
        if let Some(v) = &vector {
            let space_id = ours.as_ref().map(|sp| sp.space_id.as_str()).unwrap_or_default();
            let theirs = s
                .ver
                .vectors
                .get(&key)
                .filter(|(sp, _)| sp == space_id)
                .map(|(_, v)| v.clone())
                // the vector lane's own vectors are in this space when its model is the one Records embed with
                .or_else(|| (space_id == lane_space).then(|| s.vera.lane.store.get(&key).map(<[f32]>::to_vec)).flatten());
            if let Some(t) = theirs
                && t.len() == v.len()
            {
                let c = lightcraft_vera::cutoff::cosine(v, &t);
                if c >= thresholds::EMBED_NEAR {
                    matches.push(Match { id: photo, by: "vector", score: (f64::from(c) * 10_000.0).round() / 10_000.0 });
                }
            }
        }
    }
    // by tier; distances ascending, cosines descending
    matches.sort_by(|a, b| {
        rank(a.by)
            .cmp(&rank(b.by))
            .then_with(|| if a.by == "vector" { b.score.total_cmp(&a.score) } else { a.score.total_cmp(&b.score) })
            .then(a.id.cmp(&b.id))
    });
    Ok(Consumed { status: "accepted", issues, matches, summary, pending })
}
