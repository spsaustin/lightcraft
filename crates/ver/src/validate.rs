//! Structural (schema) and profile (cross-object) validation, coded as the upstream conformance
//! validator (`packages/ver-validator`, profile `vera-profile/1.0`) codes them.
//!
//! [`structural`] runs the `jsonschema` crate (JSON Schema 2020-12) against the schema bytes
//! vendored in `standard/schemas/` and reports what the upstream schema stage reports:
//! `VER102` (not an object), `VER201` (schema-invalid, `format` left an annotation) and `VER202`
//! (errata E23: `format` asserted — `date-time`, `date`, `uri`, `uuid` with the upstream
//! checker's exact shapes — and `metadata.raw[].bytes` decodable as strict base64).
//!
//! [`profile`] implements the cross-object rules listed in [`PROFILE_CODES`], each under its
//! upstream code and severity; [`LOCAL_CODES`] are rules of ours that the upstream profile has no
//! code for.

use crate::record::Schema;
use crate::time;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

/// How bad an issue is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub enum Severity {
    Error,
    Warning,
}

/// One finding: `code` is the upstream code (`VER101`…), or a [`LOCAL_CODES`] entry.
#[derive(Clone, Debug, Serialize)]
pub struct Issue {
    pub code: String,
    pub severity: Severity,
    pub pointer: String,
    pub message: String,
}

/// The upstream codes [`structural`] can emit.
pub const STRUCTURAL_CODES: &[&str] = &["VER102", "VER201", "VER202"];

/// The upstream codes [`profile`] implements (severity as the upstream registry has it).
pub const PROFILE_CODES: &[&str] = &[
    "VER202", "VER301", "VER302", "VER303", "VER304", "VER305", "VER306", "VER307", "VER308", "VER401", "VER403", "VER404", "VER405", "VER501",
    "VER601", "VER602", "VER603", "VER604", "VER605", "VER606", "VER607", "VER608", "VER609", "VER701", "VER702", "VER703", "VER704", "VER705",
    "VER801", "VER802", "VER901", "VER1101", "VER1102", "VER1103", "VER1201", "VER1202", "VER1203", "VER1301", "VER1302", "VER1305",
];

/// Rules of ours with no upstream code: `LCVER-E2` (errata E2: `identity.pixel_hash.width/height`
/// MUST equal `image.width/height`) and `LCVER-UTC` (E23: timestamps SHOULD be UTC with `Z`).
pub const LOCAL_CODES: &[&str] = &["LCVER-E2", "LCVER-UTC"];

fn issue(code: &str, severity: Severity, pointer: String, message: impl Into<String>) -> Issue {
    Issue { code: code.into(), severity, pointer, message: message.into() }
}

fn err(code: &str, pointer: String, message: impl Into<String>) -> Issue {
    issue(code, Severity::Error, pointer, message)
}

fn warn(code: &str, pointer: String, message: impl Into<String>) -> Issue {
    issue(code, Severity::Warning, pointer, message)
}

/// An RFC 6901 pointer from path segments.
pub fn pointer(parts: &[&dyn std::fmt::Display]) -> String {
    parts.iter().map(|p| format!("/{}", p.to_string().replace('~', "~0").replace('/', "~1"))).collect()
}

macro_rules! ptr {
    ($($p:expr),* $(,)?) => { pointer(&[$(&$p as &dyn std::fmt::Display),*]) };
}

// ---------------------------------------------------------------------------------------------
// Structural
// ---------------------------------------------------------------------------------------------

const SCHEMA_1_0_1: &str = include_str!("../standard/schemas/ver-record-1.0.1.schema.json");
const SCHEMA_1_1_DRAFT: &str = include_str!("../standard/schemas/ver-record-1.1-draft.schema.json");

/// The vendored schema bytes for `schema` (byte-identical to the upstream release files).
pub fn schema_source(schema: Schema) -> &'static str {
    match schema {
        Schema::V1_0_1 => SCHEMA_1_0_1,
        Schema::V1_1Draft => SCHEMA_1_1_DRAFT,
    }
}

fn check_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 36
        && b.iter().enumerate().all(|(i, &c)| match i {
            8 | 13 | 18 | 23 => c == b'-',
            _ => c.is_ascii_digit() || (b'a'..=b'f').contains(&c),
        })
}

fn check_uri(s: &str) -> bool {
    let Some((scheme, rest)) = s.split_once(':') else { return false };
    let mut sc = scheme.bytes();
    if !sc.next().is_some_and(|c| c.is_ascii_alphabetic()) || !sc.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'.' | b'-')) {
        return false;
    }
    let ok_char = |c: u8| c.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=:@/?#[]%".contains(&c);
    let b = rest.as_bytes();
    if !b.iter().all(|&c| ok_char(c)) {
        return false;
    }
    b.iter()
        .enumerate()
        .all(|(i, &c)| c != b'%' || (b.get(i + 1).is_some_and(u8::is_ascii_hexdigit) && b.get(i + 2).is_some_and(u8::is_ascii_hexdigit)))
}

fn check_date_time(s: &str) -> bool {
    time::parse(s).is_some()
}

struct Compiled {
    plain: Result<jsonschema::Validator, String>,
    formats: Result<jsonschema::Validator, String>,
}

fn compile(schema: Schema) -> Compiled {
    let parsed: Result<Value, String> = serde_json::from_str(schema_source(schema)).map_err(|e| e.to_string());
    let build = |formats: bool| -> Result<jsonschema::Validator, String> {
        let v = parsed.as_ref().map_err(Clone::clone)?;
        let mut o = jsonschema::draft202012::options().should_validate_formats(formats);
        if formats {
            o = o
                .with_format("uuid", check_uuid)
                .with_format("uri", check_uri)
                .with_format("date-time", check_date_time)
                .with_format("date", time::valid_date);
        }
        o.build(v).map_err(|e| e.to_string())
    };
    Compiled { plain: build(false), formats: build(true) }
}

fn compiled(schema: Schema) -> &'static Compiled {
    static V101: OnceLock<Compiled> = OnceLock::new();
    static V11: OnceLock<Compiled> = OnceLock::new();
    match schema {
        Schema::V1_0_1 => V101.get_or_init(|| compile(Schema::V1_0_1)),
        Schema::V1_1Draft => V11.get_or_init(|| compile(Schema::V1_1Draft)),
    }
}

/// Cap on reported `VER201` issues per document (the upstream cap).
const MAX_SCHEMA_ISSUES: usize = 50;

/// Validate `doc` against the vendored schema: `VER102`, `VER201`, `VER202`.
pub fn structural(doc: &Value, schema: Schema) -> Vec<Issue> {
    if !doc.is_object() {
        return vec![err("VER102", String::new(), "a VER Record is a JSON object")];
    }
    let c = compiled(schema);
    let mut out = Vec::new();
    match &c.plain {
        Ok(v) => {
            let mut seen = BTreeSet::new();
            for e in v.iter_errors(doc) {
                let p = e.instance_path().to_string();
                let m = e.to_string();
                if seen.insert((p.clone(), m.clone())) {
                    out.push(err("VER201", p, m));
                }
                if out.len() >= MAX_SCHEMA_ISSUES {
                    break;
                }
            }
        }
        Err(e) => out.push(err("VER201", String::new(), format!("vendored schema does not compile: {e}"))),
    }
    if let Ok(v) = &c.formats {
        let mut seen = BTreeSet::new();
        for e in v.iter_errors(doc) {
            if matches!(e.kind(), jsonschema::error::ValidationErrorKind::Format { .. }) {
                let p = e.instance_path().to_string();
                let m = format!("{e} (JSON Schema 2020-12 treats format as an annotation; the profile asserts it, errata E23)");
                if seen.insert(p.clone()) {
                    out.push(err("VER202", p, m));
                }
            }
        }
    }
    out.extend(content_encoding(doc));
    out
}

fn content_encoding(doc: &Value) -> Vec<Issue> {
    let mut out = Vec::new();
    for (i, entry) in arr(doc.pointer("/metadata/raw")).iter().enumerate() {
        if let Some(b) = entry.get("bytes").and_then(Value::as_str)
            && crate::b64::decode(b).is_none()
        {
            out.push(err(
                "VER202",
                ptr!("metadata", "raw", i, "bytes"),
                "contentEncoding base64 is not decodable (RFC 4648 §4: standard alphabet, padded, unwrapped)",
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------------------------

fn arr(v: Option<&Value>) -> &[Value] {
    v.and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn s<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}

fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_i64)
}

const FAMILIES: [&str; 5] = ["exif", "xmp", "iptc", "icc", "c2pa"];

fn segment_family(family: &str) -> &str {
    if family == "iptc" { "iptc-iim" } else { family }
}

fn level_rank(l: &str) -> Option<u8> {
    match l {
        "L0" => Some(0),
        "L1" => Some(1),
        "L2" => Some(2),
        "L3" => Some(3),
        _ => None,
    }
}

const SPACE_ID_PATTERN_HELP: &str = "reverse-DNS authority + '/' + semver";

/// `[a-z0-9]+(\.[a-z0-9-]+)+/[0-9]+\.[0-9]+(\.[0-9]+)?`, full match.
fn space_id_ok(id: &str) -> bool {
    let Some((auth, ver)) = id.split_once('/') else { return false };
    let mut labels = auth.split('.');
    let first_ok = labels.next().is_some_and(|l| !l.is_empty() && l.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    let rest: Vec<&str> = labels.collect();
    let rest_ok =
        !rest.is_empty() && rest.iter().all(|l| !l.is_empty() && l.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-'));
    let parts: Vec<&str> = ver.split('.').collect();
    let ver_ok = (parts.len() == 2 || parts.len() == 3) && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit()));
    first_ok && rest_ok && ver_ok
}

fn byte_len(dim: i64, dtype: &str) -> Option<i64> {
    if dim <= 0 {
        return None;
    }
    match dtype {
        "binary" => Some((dim + 7) / 8),
        "fp32" => Some(dim * 4),
        "fp16" | "bf16" => Some(dim * 2),
        "int8" => Some(dim),
        _ => None,
    }
}

/// The cross-object profile rules ([`PROFILE_CODES`], [`LOCAL_CODES`]), at the level the
/// document claims (`L0` when it claims none, with `VER301`).
pub fn profile(doc: &Value) -> Vec<Issue> {
    profile_at(doc, None)
}

/// [`profile`] with the effective level forced (the upstream CLI's `--level`): the claim is still
/// read (and `VER301` still reported when absent), but `level` decides which gates apply.
pub fn profile_at(doc: &Value, level: Option<crate::record::Level>) -> Vec<Issue> {
    let mut out = Vec::new();
    if !doc.is_object() {
        out.push(err("VER102", String::new(), "a VER Record is a JSON object"));
        return out;
    }
    let draft = s(doc, "ver_version") == Some("1.1");

    // Stage 3: level.
    let claimed = s(doc, "conformance_level").filter(|l| level_rank(l).is_some());
    if claimed.is_none() {
        out.push(warn("VER301", ptr!("conformance_level"), "conformance_level is absent or not one of L0..L3; validating at L0"));
    }
    let level = match level {
        Some(l) => l as u8,
        None => claimed.and_then(level_rank).unwrap_or(0),
    };
    let avail = doc.pointer("/metadata/availability");
    let state = |f: &str| avail.and_then(|a| a.get(f)).and_then(Value::as_str);
    let raw = arr(doc.pointer("/metadata/raw"));
    let preserved: BTreeMap<&str, usize> = raw.iter().enumerate().filter_map(|(i, e)| s(e, "segment").map(|f| (f, i))).rev().collect();
    if level >= 1 {
        for f in FAMILIES {
            if state(f) == Some("present") && !preserved.contains_key(segment_family(f)) {
                out.push(err(
                    "VER302",
                    ptr!("metadata", "availability", f),
                    format!("family {f:?} is declared present, but metadata.raw preserves no {:?} segment (errata E11)", segment_family(f)),
                ));
            }
        }
        let missing: Vec<&str> = FAMILIES.iter().copied().filter(|f| state(f).is_none()).collect();
        if !missing.is_empty() {
            out.push(err(
                "VER303",
                ptr!("metadata", "availability"),
                format!("L1+ requires an availability state for all five families; missing: {}", missing.join(", ")),
            ));
        }
    }
    if level >= 2 {
        let normalized = doc.pointer("/metadata/normalized").and_then(Value::as_object);
        let assertions: Vec<&String> = normalized.map(|n| n.keys().filter(|k| !k.starts_with('@')).collect()).unwrap_or_default();
        if assertions.is_empty() {
            out.push(err(
                "VER304",
                ptr!("metadata", "normalized"),
                "L2 requires a reconciled metadata.normalized view carrying at least one assertion",
            ));
        } else {
            let tracked: BTreeSet<&str> = arr(doc.pointer("/metadata/trust")).iter().filter_map(|t| s(t, "field")).collect();
            for a in assertions {
                if !tracked.contains(a.as_str()) {
                    out.push(err("VER305", ptr!("metadata", "normalized", a), format!("normalized assertion {a:?} has no metadata.trust entry")));
                }
            }
        }
    }
    if level >= 3 {
        if doc.pointer("/provenance/acquisition").and_then(Value::as_object).is_none() {
            out.push(err("VER306", ptr!("provenance", "acquisition"), "L3 requires provenance.acquisition"));
        }
        if arr(doc.pointer("/provenance/chain")).is_empty() {
            out.push(err("VER306", ptr!("provenance", "chain"), "L3 requires a non-empty provenance.chain"));
        }
        if doc.pointer("/provenance/c2pa/status").and_then(Value::as_str).is_none() {
            out.push(err("VER307", ptr!("provenance", "c2pa", "status"), "L3 requires provenance.c2pa.status, including the honest 'missing' state"));
        }
        if doc.pointer("/provenance/signature").and_then(Value::as_object).is_none() {
            out.push(warn("VER308", ptr!("provenance", "signature"), "L3 records SHOULD carry provenance.signature; none is present"));
        }
    }

    // Stage 4: spaces.
    let spaces = arr(doc.get("spaces"));
    let mut declared: BTreeMap<&str, &Value> = BTreeMap::new();
    for (i, d) in spaces.iter().enumerate() {
        if !d.is_object() {
            continue;
        }
        if let Some(id) = s(d, "space_id") {
            if declared.contains_key(id) {
                out.push(err("VER401", ptr!("spaces", i, "space_id"), format!("space_id {id:?} is declared more than once (errata E18)")));
            } else {
                declared.insert(id, d);
            }
        }
        if d.get("reference").and_then(Value::as_object).is_none() {
            out.push(warn(
                "VER404",
                ptr!("spaces", i, "reference"),
                "space descriptor carries no reference block; it cannot back a conformance claim (errata E13)",
            ));
        }
        let required: &[&str] = match s(d, "kind") {
            Some("visual") => &["resize", "interpolation", "antialias", "crop", "mean", "std", "channel_order"],
            Some("text") => &["tokenizer", "tokenizer_sha256", "max_length", "truncation", "casing"],
            _ => &[],
        };
        let pre = d.get("preprocessing").and_then(Value::as_object);
        let missing: Vec<&str> = required.iter().copied().filter(|k| !pre.is_some_and(|p| p.contains_key(*k))).collect();
        if !missing.is_empty() {
            out.push(warn("VER405", ptr!("spaces", i, "preprocessing"), format!("preprocessing omits {} (errata E13)", missing.join(", "))));
        }
    }

    // Stages 5–7: references, vectors, roles.
    let embeddings = arr(doc.get("embeddings"));
    let mut has_canonical = false;
    for (i, e) in embeddings.iter().enumerate() {
        if !e.is_object() {
            continue;
        }
        let space_id = s(e, "space_id");
        if let Some(id) = space_id {
            if !space_id_ok(id) {
                out.push(err(
                    "VER403",
                    ptr!("embeddings", i, "space_id"),
                    format!("embedding space_id {id:?} does not match the descriptor pattern ({SPACE_ID_PATTERN_HELP})"),
                ));
            }
            if !declared.contains_key(id) {
                out.push(err(
                    "VER501",
                    ptr!("embeddings", i, "space_id"),
                    format!("embedding references space_id {id:?}, which is not declared in spaces[]"),
                ));
            }
        }
        let desc = space_id.and_then(|id| declared.get(id)).copied();
        // VER607.
        let vector = e.get("vector").and_then(Value::as_array);
        if let Some(v) = vector
            && let Some(first) = v.iter().position(|c| c.as_f64().is_some_and(|f| !f.is_finite()))
        {
            out.push(err("VER607", ptr!("embeddings", i, "vector", first), "inline vector component is not a finite number"));
        }
        // VER601/602/606.
        let own_dim = int(e.get("dim"));
        let own_dtype = s(e, "dtype");
        let desc_dim = desc.and_then(|d| int(d.get("dim")));
        let desc_dtype = desc.and_then(|d| s(d, "dtype"));
        if let (Some(a), Some(b)) = (own_dim, desc_dim)
            && a != b
        {
            out.push(err("VER601", ptr!("embeddings", i, "dim"), format!("embedding dim {a} does not equal the descriptor dim {b} (errata E24)")));
        }
        if let (Some(a), Some(b)) = (own_dtype, desc_dtype)
            && a != b
        {
            out.push(err("VER602", ptr!("embeddings", i, "dtype"), format!("embedding dtype {a:?} does not equal the descriptor dtype {b:?}")));
        }
        if own_dim.is_none() || own_dtype.is_none() {
            out.push(warn(
                "VER606",
                ptr!("embeddings", i),
                "embedding omits dim and/or dtype; the descriptor's value applies (mandatory in 1.1, ADR-0008)",
            ));
        }
        let dim = own_dim.or(desc_dim);
        let dtype = own_dtype.or(desc_dtype);
        if let (Some(dim), Some(dtype)) = (dim, dtype) {
            if let Some(v) = vector
                && v.len() as i64 != dim
            {
                out.push(err(
                    "VER603",
                    ptr!("embeddings", i, "vector"),
                    format!("inline vector carries {} components but the space has dim {dim}", v.len()),
                ));
            }
            let expected = byte_len(dim, dtype);
            if let (Some(b64), Some(expected)) = (s(e, "vector_b64"), expected)
                && let Some(bytes) = crate::b64::decode(b64)
                && bytes.len() as i64 != expected
            {
                out.push(err(
                    "VER604",
                    ptr!("embeddings", i, "vector_b64"),
                    format!("vector_b64 decodes to {} bytes; dim {dim} at dtype {dtype} requires {expected}", bytes.len()),
                ));
            }
            if let (Some(len), Some(expected)) = (int(e.pointer("/vector_ref/byte_length")), expected)
                && len != expected
            {
                out.push(err(
                    "VER605",
                    ptr!("embeddings", i, "vector_ref", "byte_length"),
                    format!("vector_ref.byte_length is {len}; dim {dim} at dtype {dtype} requires {expected}"),
                ));
            }
        }
        if draft {
            let forms: Vec<&str> = ["vector", "vector_b64", "vector_ref"].into_iter().filter(|f| e.get(*f).is_some()).collect();
            if forms.len() != 1 {
                out.push(err("VER608", ptr!("embeddings", i), "an embedding carries exactly one of vector, vector_b64, vector_ref"));
            }
            if dtype == Some("binary") && vector.is_some() {
                out.push(err("VER609", ptr!("embeddings", i, "vector"), "a binary vector has no inline JSON array carriage (ADR-0008)"));
            }
        }
        // Stage 7: roles.
        let role = s(e, "role");
        if role == Some("canonical_visual") {
            has_canonical = true;
        }
        if let (Some(role), Some(d)) = (role, desc)
            && let Some(kind) = s(d, "kind")
        {
            let (wanted, code): (&[&str], &str) = match role {
                "canonical_visual" if draft => (&["visual", "joint_visual_text"], "VER701"),
                "canonical_visual" => (&["visual"], "VER701"),
                "contextual" if draft => (&["text", "joint_visual_text"], "VER702"),
                "contextual" => (&["text"], "VER702"),
                "fused" => (&["fused"], "VER703"),
                _ => (&[], ""),
            };
            if !wanted.is_empty() && !wanted.contains(&kind) {
                out.push(err(
                    code,
                    ptr!("embeddings", i, "space_id"),
                    format!("embedding role {role:?} references a space of kind {kind:?}; it requires {}", wanted.join(" or ")),
                ));
            }
        }
        if role == Some("fused") {
            match e.get("recipe").and_then(Value::as_object) {
                None => out.push(err("VER704", ptr!("embeddings", i, "recipe"), "fused embedding carries no recipe")),
                Some(r) => {
                    let inputs = r.get("inputs").and_then(Value::as_array);
                    let distinct: BTreeSet<&str> = inputs.map(|v| v.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                    if inputs.is_none_or(|v| v.len() < 2) || distinct.len() < 2 {
                        out.push(err(
                            "VER704",
                            ptr!("embeddings", i, "recipe", "inputs"),
                            "recipe.inputs must name at least two distinct spaces (errata E20)",
                        ));
                    }
                    if let Some(v) = inputs
                        && v.iter().any(|x| x.as_str().is_none_or(|id| !declared.contains_key(id)))
                    {
                        out.push(err(
                            "VER704",
                            ptr!("embeddings", i, "recipe", "inputs"),
                            "recipe.inputs names a space the record does not declare in spaces[]",
                        ));
                    }
                }
            }
        }
        // Stage 14: vector_b64 decodability.
        if let Some(b64) = s(e, "vector_b64")
            && crate::b64::decode(b64).is_none()
        {
            out.push(err("VER1301", ptr!("embeddings", i, "vector_b64"), "vector_b64 is not decodable base64"));
        }
    }
    if !has_canonical {
        out.push(err("VER705", ptr!("embeddings"), "no embedding carries role 'canonical_visual'"));
    }

    // Stage 8: perceptual digests.
    for (i, p) in arr(doc.pointer("/identity/perceptual")).iter().enumerate() {
        let (Some(alg), Some(value)) = (s(p, "alg"), s(p, "value")) else { continue };
        let (width, code) = match alg {
            "pdq/1.0" => (64, "VER801"),
            "phash-dct/64" => (16, "VER802"),
            _ => continue,
        };
        let hex = value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c));
        if value.len() != width || !hex {
            out.push(err(
                code,
                ptr!("identity", "perceptual", i, "value"),
                format!("{alg} digest must be exactly {width} lowercase hex characters (errata E22)"),
            ));
        }
    }

    // E2 (local): pixel_hash dimensions are the image's.
    let (pw, ph) = (int(doc.pointer("/identity/pixel_hash/width")), int(doc.pointer("/identity/pixel_hash/height")));
    let (iw, ih) = (int(doc.pointer("/image/width")), int(doc.pointer("/image/height")));
    if let (Some(pw), Some(ph), Some(iw), Some(ih)) = (pw, ph, iw, ih)
        && (pw, ph) != (iw, ih)
    {
        out.push(err(
            "LCVER-E2",
            ptr!("identity", "pixel_hash"),
            format!("identity.pixel_hash is {pw}×{ph} but image is {iw}×{ih}; errata E2 requires them equal"),
        ));
    }

    // Stage 10: availability basis.
    let suspected: Vec<&str> = FAMILIES.iter().copied().chain(draft.then_some("other")).filter(|f| state(f) == Some("stripped_suspected")).collect();
    if !suspected.is_empty() {
        let basis = doc.pointer("/metadata/availability_basis");
        let stated = |v: Option<&Value>| v.and_then(Value::as_str).is_some_and(|t| !t.trim().is_empty());
        let missing: Vec<&str> = if stated(basis) {
            Vec::new()
        } else if draft && basis.is_some_and(Value::is_object) {
            suspected.iter().copied().filter(|f| !stated(basis.and_then(|b| b.get(*f)))).collect()
        } else {
            suspected.clone()
        };
        if !missing.is_empty() {
            out.push(err(
                "VER901",
                ptr!("metadata", "availability_basis"),
                format!("{} declared 'stripped_suspected' with no stated basis (errata E21)", missing.join(", ")),
            ));
        }
    }

    // Stage 12: C2PA consistency (errata E28).
    let c2pa = doc.pointer("/provenance/c2pa").filter(|v| v.is_object());
    let status = c2pa.and_then(|c| s(c, "status"));
    let raw_c2pa = preserved.get("c2pa").copied();
    let avail_c2pa = state("c2pa");
    let evidence = raw_c2pa.is_some() || avail_c2pa == Some("present");
    if evidence && (status == Some("missing") || (c2pa.is_none() && level >= 3)) {
        out.push(err(
            "VER1101",
            ptr!("provenance", "c2pa", "status"),
            "provenance.c2pa.status is missing/absent while the record carries C2PA evidence",
        ));
    }
    let contradiction = status == Some("missing") && evidence;
    let absence = status.is_none() && avail_c2pa == Some("present") && level >= 3;
    if contradiction || absence {
        out.push(err("VER1103", ptr!("provenance", "c2pa"), "C2PA state is internally inconsistent (errata E28)"));
    }
    if status == Some("valid") && c2pa.and_then(|c| s(c, "manifest_sha256")).is_none() {
        out.push(warn("VER1102", ptr!("provenance", "c2pa"), "provenance.c2pa.status is 'valid' but no manifest_sha256 is recorded"));
    }

    // Stage 13: chronology (errata E16), and the profile's timestamp assertions.
    let chain = arr(doc.pointer("/provenance/chain"));
    let moments: Vec<(usize, time::Instant)> =
        chain.iter().enumerate().filter_map(|(i, ev)| s(ev, "at").and_then(time::parse).map(|(t, _)| (i, t))).collect();
    for w in moments.windows(2) {
        if let [(_, a), (j, b)] = w
            && b < a
        {
            out.push(err("VER1201", ptr!("provenance", "chain", j, "at"), "chain timestamps must be non-decreasing"));
        }
    }
    if let Some((acq, _)) = doc.pointer("/provenance/acquisition/acquired_at").and_then(Value::as_str).and_then(time::parse) {
        for (i, t) in &moments {
            if *t < acq {
                out.push(err("VER1202", ptr!("provenance", "chain", i, "at"), "chain event precedes acquisition"));
            }
        }
    }
    if let (Some((created, _)), Some(latest)) = (s(doc, "created_at").and_then(time::parse), moments.iter().map(|(_, t)| *t).max())
        && created < latest
    {
        out.push(warn("VER1203", ptr!("created_at"), "created_at precedes the last chain event (errata E16)"));
    }
    let mut stamps: Vec<(String, &str)> = Vec::new();
    if let Some(t) = s(doc, "created_at") {
        stamps.push((ptr!("created_at"), t));
    }
    for (i, e) in embeddings.iter().enumerate() {
        if let Some(t) = s(e, "produced_at") {
            stamps.push((ptr!("embeddings", i, "produced_at"), t));
        }
    }
    for (p, t) in stamps {
        match time::parse(t) {
            None => out.push(err("VER202", p, format!("{t:?} is not an RFC 3339 date-time (errata E23)"))),
            Some((_, false)) => {
                out.push(warn("LCVER-UTC", p, format!("{t:?} is not UTC; producers SHOULD emit UTC with a Z designator (errata E23)")))
            }
            Some(_) => {}
        }
    }
    if let Some(id) = s(doc, "record_id")
        && !check_uuid(id)
    {
        out.push(err("VER202", ptr!("record_id"), format!("{id:?} is not a canonical lowercase UUID (errata E23)")));
    }

    // Stage 14: checksums over inline bytes.
    for (i, entry) in raw.iter().enumerate() {
        let (Some(payload), Some(claimed)) = (s(entry, "bytes"), s(entry, "sha256")) else { continue };
        let Some(bytes) = crate::b64::decode(payload) else { continue };
        let actual = crate::hashing::sha256_hex(&bytes);
        if actual != claimed.to_ascii_lowercase() {
            out.push(err(
                "VER1302",
                ptr!("metadata", "raw", i, "sha256"),
                format!("raw segment declares sha256 {claimed}, but the carried bytes hash to {actual}"),
            ));
        }
    }
    if let (Some(r), Some(c)) =
        (doc.pointer("/image/source_bytes_ref/sha256").and_then(Value::as_str), doc.pointer("/identity/content_hash/value").and_then(Value::as_str))
        && !r.eq_ignore_ascii_case(c)
    {
        out.push(err(
            "VER1305",
            ptr!("image", "source_bytes_ref", "sha256"),
            "image.source_bytes_ref.sha256 does not equal identity.content_hash.value (errata E25)",
        ));
    }
    out
}

/// [`structural`] then [`profile`], without repeating a `(code, pointer)` pair both report.
pub fn all(doc: &Value, schema: Schema) -> Vec<Issue> {
    let mut v = structural(doc, schema);
    let seen: BTreeSet<(String, String)> = v.iter().map(|i| (i.code.clone(), i.pointer.clone())).collect();
    v.extend(profile(doc).into_iter().filter(|i| !seen.contains(&(i.code.clone(), i.pointer.clone()))));
    v
}

/// No Error-severity issue.
pub fn is_conformant(issues: &[Issue]) -> bool {
    !issues.iter().any(|i| i.severity == Severity::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_checks() {
        assert!(check_uuid("a3f1c2e4-9b7d-4e21-8c55-2f6a0d9b3e71"));
        assert!(!check_uuid("A3F1C2E4-9B7D-4E21-8C55-2F6A0D9B3E71"));
        assert!(!check_uuid("not-a-uuid"));
        assert!(check_uri("https://example.org/a?b=c#d"));
        assert!(!check_uri("not a uri at all ::::"));
        assert!(!check_uri("https://example.org/%zz"));
        assert!(space_id_ok("org.verstandard.space.clip-vit-b32/1.0"));
        assert!(!space_id_ok("Org.x/1.0"));
        assert!(!space_id_ok("org/1.0"));
    }
}
