//! VER Records (artifact release 1.0.1 and the 1.1 draft): a builder that refuses to emit a
//! document that does not validate, and a parser.

use crate::cpnp::Format;
use crate::hashing::Identity;
use crate::segments::{Availability, RawSegment};
use crate::validate;
use serde_json::{Map, Value, json};

/// Which schema a Record targets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Schema {
    V1_0_1,
    V1_1Draft,
}

impl Schema {
    /// The `ver_version` value records under this schema carry.
    pub fn ver_version(self) -> &'static str {
        match self {
            Schema::V1_0_1 => "1.0",
            Schema::V1_1Draft => "1.1",
        }
    }

    /// The schema for a `ver_version` value.
    pub fn from_ver_version(v: &str) -> Option<Schema> {
        match v {
            "1.0" => Some(Schema::V1_0_1),
            "1.1" => Some(Schema::V1_1Draft),
            _ => None,
        }
    }
}

/// A conformance level.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Level {
    L0,
    L1,
    L2,
    L3,
}

impl Level {
    /// The schema token.
    pub fn as_str(self) -> &'static str {
        match self {
            Level::L0 => "L0",
            Level::L1 => "L1",
            Level::L2 => "L2",
            Level::L3 => "L3",
        }
    }
}

/// A Record's identity, as read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentityView {
    pub content_hash: String,
    pub pixel_hash: String,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub pdq: Option<String>,
    pub phash: Option<String>,
}

/// One embedding as read back: `(role, space_id, dim, dtype, vector)`. `dim` and `dtype` inherit
/// from the space descriptor when the embedding omits them (errata E24); `vector` is decoded from
/// `vector` or `vector_b64` (little-endian `dtype`), `None` for `vector_ref` or a payload that
/// does not decode.
pub type EmbeddingEntry = (String, String, Option<usize>, Option<String>, Option<Vec<f32>>);

/// A complete VER Record document.
#[derive(Clone, Debug, PartialEq)]
pub struct Record(pub Value);

impl Record {
    /// Pretty-printed JSON.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.0).unwrap_or_default()
    }

    /// Parse a document (no validation beyond "a JSON object"; see [`crate::validate`]).
    pub fn parse(s: &str) -> Result<Record, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("VER101 unparseable JSON: {e}"))?;
        if !v.is_object() {
            return Err("VER102 a VER Record is a JSON object".into());
        }
        Ok(Record(v))
    }

    /// The schema its `ver_version` selects.
    pub fn schema(&self) -> Option<Schema> {
        self.0.get("ver_version").and_then(Value::as_str).and_then(Schema::from_ver_version)
    }

    /// `identity`, when its two required hashes are present.
    pub fn identity(&self) -> Option<IdentityView> {
        let id = self.0.get("identity")?;
        let perceptual = |alg: &str| {
            id.get("perceptual")?
                .as_array()?
                .iter()
                .find(|p| p.get("alg").and_then(Value::as_str) == Some(alg))?
                .get("value")?
                .as_str()
                .map(str::to_string)
        };
        Some(IdentityView {
            content_hash: id.pointer("/content_hash/value")?.as_str()?.to_string(),
            pixel_hash: id.pointer("/pixel_hash/value")?.as_str()?.to_string(),
            width: id.pointer("/pixel_hash/width").and_then(Value::as_u64),
            height: id.pointer("/pixel_hash/height").and_then(Value::as_u64),
            pdq: perceptual("pdq/1.0"),
            phash: perceptual("phash-dct/64"),
        })
    }

    /// Every embedding, decoded where it is carried inline.
    pub fn embeddings(&self) -> Vec<EmbeddingEntry> {
        let spaces = self.spaces();
        let descriptor = |id: &str| spaces.iter().find(|d| d.get("space_id").and_then(Value::as_str) == Some(id)).cloned();
        let mut out = Vec::new();
        for e in self.0.get("embeddings").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
            let role = e.get("role").and_then(Value::as_str).unwrap_or_default().to_string();
            let space_id = e.get("space_id").and_then(Value::as_str).unwrap_or_default().to_string();
            let desc = descriptor(&space_id);
            let dim = e.get("dim").or_else(|| desc.as_ref()?.get("dim")).and_then(Value::as_u64).and_then(|d| usize::try_from(d).ok());
            let dtype = e.get("dtype").or_else(|| desc.as_ref()?.get("dtype")).and_then(Value::as_str).map(str::to_string);
            let vector = if let Some(v) = e.get("vector").and_then(Value::as_array) {
                v.iter().map(|c| c.as_f64().map(|f| f as f32)).collect::<Option<Vec<f32>>>()
            } else if let (Some(b64), Some(dt)) = (e.get("vector_b64").and_then(Value::as_str), dtype.as_deref()) {
                crate::b64::decode(b64).map(|bytes| {
                    let mut v = unpack_vector(&bytes, dt);
                    if let Some(d) = dim {
                        v.truncate(d);
                    }
                    v
                })
            } else {
                None
            };
            out.push((role, space_id, dim, dtype, vector));
        }
        out
    }

    /// The `spaces[]` descriptors.
    pub fn spaces(&self) -> Vec<Value> {
        self.0.get("spaces").and_then(Value::as_array).cloned().unwrap_or_default()
    }
}

/// Little-endian packing of `v` as `dtype` (errata E24): `fp32` 4 octets, `fp16` IEEE binary16
/// (round to nearest even, as numpy's `astype("<f2")`), `bf16` the high half of fp32 rounded to
/// nearest even, `int8` each component rounded and clamped to −128..=127, `binary` one bit per
/// component (`> 0`), eight per octet, most significant bit first. Unknown dtypes pack nothing.
pub fn pack_vector(v: &[f32], dtype: &str) -> Vec<u8> {
    match dtype {
        "fp32" => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        "fp16" => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        "bf16" => v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect(),
        "int8" => v.iter().map(|x| (x.round().clamp(-128.0, 127.0) as i8) as u8).collect(),
        "binary" => v.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |acc, (i, x)| if *x > 0.0 { acc | (0x80 >> i) } else { acc })).collect(),
        _ => Vec::new(),
    }
}

/// The inverse of [`pack_vector`] (`binary` yields eight 0/1 components per octet; the caller
/// truncates to `dim`).
pub fn unpack_vector(bytes: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "fp32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "fp16" => bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "bf16" => bytes.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        "int8" => bytes.iter().map(|&b| f32::from(b as i8)).collect(),
        "binary" => bytes.iter().flat_map(|b| (0..8).map(move |i| if b & (0x80 >> i) != 0 { 1.0 } else { 0.0 })).collect(),
        _ => Vec::new(),
    }
}

/// `v` scaled to unit L2 norm in `f32` (unchanged when its norm is zero or not finite).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 && n.is_finite() { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}

/// An RFC 9562 UUIDv5 (SHA-1, name-based) of `name` in `namespace`, lowercase hyphenated.
pub fn uuid_v5(namespace: &[u8; 16], name: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    let mut h = Sha1::new();
    h.update(namespace);
    h.update(name);
    let d = h.finalize();
    let mut b = [0u8; 16];
    b.copy_from_slice(&d[..16]);
    b[6] = (b[6] & 0x0F) | 0x50;
    b[8] = (b[8] & 0x3F) | 0x80;
    let hex = crate::hashing::hex(&b);
    format!("{}-{}-{}-{}-{}", &hex[0..8], &hex[8..12], &hex[12..16], &hex[16..20], &hex[20..32])
}

/// The namespace LightCraft derives default `record_id`s in (a UUIDv5 of the URL
/// `https://github.com/storytold/lightcraft/ver` in the RFC 9562 URL namespace).
pub fn lightcraft_namespace() -> [u8; 16] {
    const URL_NS: [u8; 16] = [0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8];
    let id = uuid_v5(&URL_NS, b"https://github.com/storytold/lightcraft/ver");
    let hex: String = id.chars().filter(|c| *c != '-').collect();
    let mut out = [0u8; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = hex.get(i * 2..i * 2 + 2).and_then(|p| u8::from_str_radix(p, 16).ok()).unwrap_or(0);
    }
    out
}

/// Errata E16's idempotent `record_id`: UUIDv5 over the producer identity and `content_hash`
/// (plus the record format, so a 1.0 and a 1.1 dual-publish do not collide).
pub fn default_record_id(generator_name: &str, content_hash: &str, schema: Schema) -> String {
    uuid_v5(&lightcraft_namespace(), format!("{generator_name}\n{content_hash}\n{}", schema.ver_version()).as_bytes())
}

#[derive(Clone, Debug)]
struct PendingEmbedding {
    role: String,
    space_id: String,
    dim: usize,
    dtype: String,
    vector: Vec<f32>,
    produced_at: String,
    producer: String,
}

/// Assembles a Record. Every setter is infallible; [`Builder::build`] reports what is wrong and
/// refuses to emit a document with any Error from [`validate::all`].
#[derive(Clone, Debug)]
pub struct Builder {
    schema: Schema,
    generator: (String, String),
    identity: Option<(Identity, Format, bool, bool, bool)>,
    source_ref: Option<Value>,
    availability: Option<Availability>,
    raw: Vec<RawSegment>,
    normalized: Option<(Value, Vec<(String, String)>)>,
    spaces: Vec<Value>,
    embeddings: Vec<PendingEmbedding>,
    acquisition: Option<Value>,
    chain: Vec<Value>,
    c2pa: Option<String>,
    level: Option<Level>,
    record_id: Option<String>,
    created_at: Option<String>,
}

fn merge(base: &mut Map<String, Value>, extra: Value) {
    if let Value::Object(m) = extra {
        for (k, v) in m {
            base.insert(k, v);
        }
    }
}

impl Builder {
    /// A builder for `schema`, naming the producing software as `generator`.
    pub fn new(schema: Schema, generator_name: &str, generator_version: &str) -> Builder {
        Builder {
            schema,
            generator: (generator_name.to_string(), generator_version.to_string()),
            identity: None,
            source_ref: None,
            availability: None,
            raw: Vec::new(),
            normalized: None,
            spaces: Vec::new(),
            embeddings: Vec::new(),
            acquisition: None,
            chain: Vec::new(),
            c2pa: None,
            level: None,
            record_id: None,
            created_at: None,
        }
    }

    /// `identity` (hashes and perceptual digests) and `image` (Canonical Buffer dimensions, E2).
    pub fn identity(mut self, id: &Identity, format: Format, animated: bool, alpha_present: bool, hdr_present: bool) -> Self {
        self.identity = Some((id.clone(), format, animated, alpha_present, hdr_present));
        self
    }

    /// `image.source_bytes_ref` (errata E25: it points at the bytes `content_hash` covers).
    pub fn source_bytes_ref(mut self, uri: &str, sha256: &str, byte_length: u64) -> Self {
        self.source_ref = Some(json!({ "uri": uri, "sha256": sha256, "byte_length": byte_length }));
        self
    }

    /// `metadata.availability`.
    pub fn availability(mut self, a: &Availability) -> Self {
        self.availability = Some(*a);
        self
    }

    /// `metadata.raw[]`, in file order (under 1.1 each entry carries its per-family `index`).
    pub fn raw_segments(mut self, segs: &[RawSegment]) -> Self {
        self.raw = segs.to_vec();
        self
    }

    /// `metadata.normalized` (IPTC Photo Metadata JSON-LD) and one `metadata.trust` entry per
    /// `(field, tier)`.
    pub fn normalized(mut self, view: Value, trust: Vec<(String, String)>) -> Self {
        self.normalized = Some((view, trust));
        self
    }

    /// One `spaces[]` descriptor, as given.
    pub fn space(mut self, descriptor: Value) -> Self {
        self.spaces.push(descriptor);
        self
    }

    /// One embedding, carried as `vector_b64`: L2-normalised in `f32` when its space declares
    /// `normalization: "l2"`, then packed as `dtype`, little-endian (the reference-sets README's
    /// packing). `dim` and `dtype` are always stated on the embedding (ADR-0008).
    pub fn embedding(mut self, role: &str, space_id: &str, dim: usize, dtype: &str, vector: &[f32], produced_at: &str, producer: &str) -> Self {
        self.embeddings.push(PendingEmbedding {
            role: role.into(),
            space_id: space_id.into(),
            dim,
            dtype: dtype.into(),
            vector: vector.to_vec(),
            produced_at: produced_at.into(),
            producer: producer.into(),
        });
        self
    }

    /// `provenance.acquisition`; `extra` (an object) may add `source_uri` and `actor`.
    pub fn acquisition(mut self, method: &str, acquired_at: &str, extra: Value) -> Self {
        let mut m = Map::new();
        m.insert("method".into(), json!(method));
        m.insert("acquired_at".into(), json!(acquired_at));
        merge(&mut m, extra);
        self.acquisition = Some(Value::Object(m));
        self
    }

    /// One `provenance.chain[]` event; `extra` (an object) may add `params_sha256` (and, under
    /// 1.1, `target` / `uri`).
    pub fn chain_event(mut self, action: &str, actor: &str, tool: &str, at: &str, extra: Value) -> Self {
        let mut m = Map::new();
        m.insert("at".into(), json!(at));
        m.insert("actor".into(), json!(actor));
        m.insert("action".into(), json!(action));
        m.insert("tool".into(), json!(tool));
        merge(&mut m, extra);
        self.chain.push(Value::Object(m));
        self
    }

    /// `provenance.c2pa.status` (`valid`, `invalid`, `missing`; 1.1 adds `unverifiable`).
    pub fn c2pa_status(mut self, status: &str) -> Self {
        self.c2pa = Some(status.to_string());
        self
    }

    /// `conformance_level` (the claim [`validate::profile`] then holds the document to).
    pub fn level(mut self, level: Level) -> Self {
        self.level = Some(level);
        self
    }

    /// `record_id` (default: [`default_record_id`]).
    pub fn record_id(mut self, uuid: &str) -> Self {
        self.record_id = Some(uuid.to_string());
        self
    }

    /// `created_at` (default: now, UTC).
    pub fn created_at(mut self, rfc3339: &str) -> Self {
        self.created_at = Some(rfc3339.to_string());
        self
    }

    /// The document, or every Error-severity finding of [`validate::all`] when it does not validate.
    pub fn build(self) -> Result<Record, String> {
        let schema = self.schema;
        let (id, format, animated, alpha, hdr) = self.identity.clone().ok_or("identity is required: call Builder::identity")?;
        if self.spaces.is_empty() {
            return Err("at least one space descriptor is required: call Builder::space".into());
        }
        let record_id = self.record_id.clone().unwrap_or_else(|| default_record_id(&self.generator.0, &id.content_hash, schema));
        let created_at = self.created_at.clone().unwrap_or_else(crate::time::now_rfc3339);

        let mut doc = Map::new();
        doc.insert("ver_version".into(), json!(schema.ver_version()));
        doc.insert("cpnp_version".into(), json!("1.0"));
        doc.insert("record_id".into(), json!(record_id));
        doc.insert("created_at".into(), json!(created_at));
        doc.insert("generator".into(), json!({ "name": self.generator.0, "version": self.generator.1 }));
        if let Some(l) = self.level {
            doc.insert("conformance_level".into(), json!(l.as_str()));
        }
        let mut perceptual = Vec::new();
        if !id.pdq.is_empty() {
            perceptual.push(json!({ "alg": "pdq/1.0", "value": id.pdq }));
        }
        if !id.phash.is_empty() {
            perceptual.push(json!({ "alg": "phash-dct/64", "value": id.phash }));
        }
        let mut identity = json!({
            "content_hash": { "alg": "sha-256", "value": id.content_hash },
            "pixel_hash": { "alg": "sha-256", "value": id.pixel_hash, "width": id.width, "height": id.height },
        });
        if !perceptual.is_empty()
            && let Some(o) = identity.as_object_mut()
        {
            o.insert("perceptual".into(), Value::Array(perceptual));
        }
        doc.insert("identity".into(), identity);
        let mut image = json!({
            "format": format.as_str(),
            "width": id.width,
            "height": id.height,
            "animated": animated,
            "alpha_present": alpha,
            "hdr_present": hdr,
        });
        if let (Some(r), Some(o)) = (self.source_ref.clone(), image.as_object_mut()) {
            o.insert("source_bytes_ref".into(), r);
        }
        doc.insert("image".into(), image);
        doc.insert("spaces".into(), Value::Array(self.spaces.clone()));

        let mut embeddings = Vec::new();
        for e in &self.embeddings {
            let desc = self.spaces.iter().find(|d| d.get("space_id").and_then(Value::as_str) == Some(e.space_id.as_str()));
            let l2 = desc.and_then(|d| d.get("normalization")).and_then(Value::as_str) == Some("l2");
            let v = if l2 { l2_normalize(&e.vector) } else { e.vector.clone() };
            let packed = pack_vector(&v, &e.dtype);
            let mut m = Map::new();
            m.insert("role".into(), json!(e.role));
            m.insert("space_id".into(), json!(e.space_id));
            m.insert("dim".into(), json!(e.dim));
            m.insert("dtype".into(), json!(e.dtype));
            m.insert("vector_b64".into(), json!(crate::b64::encode(&packed)));
            if !e.produced_at.is_empty() {
                m.insert("produced_at".into(), json!(e.produced_at));
            }
            if !e.producer.is_empty() {
                m.insert("producer".into(), json!(e.producer));
            }
            embeddings.push(Value::Object(m));
        }
        doc.insert("embeddings".into(), Value::Array(embeddings));

        let mut metadata = Map::new();
        if !self.raw.is_empty() {
            let mut entries = crate::segments::raw_entries(&self.raw);
            if schema == Schema::V1_1Draft {
                let mut counters: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
                for (entry, seg) in entries.iter_mut().zip(&self.raw) {
                    let n = counters.entry(seg.family).or_insert(0);
                    if let Some(o) = entry.as_object_mut() {
                        o.insert("index".into(), json!(*n));
                    }
                    *n += 1;
                }
            }
            metadata.insert("raw".into(), Value::Array(entries));
        }
        if let Some(a) = &self.availability {
            let mut av = a.to_json();
            if schema == Schema::V1_1Draft
                && let Some(o) = av.as_object_mut()
            {
                let other = if self.raw.iter().any(|s| s.family == "other") { "present" } else { "absent" };
                o.insert("other".into(), json!(other));
            }
            metadata.insert("availability".into(), av);
        }
        if let Some((view, trust)) = &self.normalized {
            metadata.insert("normalized".into(), view.clone());
            metadata.insert("trust".into(), Value::Array(trust.iter().map(|(f, t)| json!({ "field": f, "tier": t })).collect()));
        }
        if !metadata.is_empty() {
            doc.insert("metadata".into(), Value::Object(metadata));
        }

        let mut provenance = Map::new();
        if let Some(a) = &self.acquisition {
            provenance.insert("acquisition".into(), a.clone());
        }
        if !self.chain.is_empty() {
            provenance.insert("chain".into(), Value::Array(self.chain.clone()));
        }
        if let Some(st) = &self.c2pa {
            provenance.insert("c2pa".into(), json!({ "status": st }));
        }
        if !provenance.is_empty() {
            doc.insert("provenance".into(), Value::Object(provenance));
        }

        let doc = Value::Object(doc);
        let errors: Vec<String> = validate::all(&doc, schema)
            .into_iter()
            .filter(|i| i.severity == validate::Severity::Error)
            .map(|i| format!("{} {} {}", i.code, i.pointer, i.message))
            .collect();
        if errors.is_empty() { Ok(Record(doc)) } else { Err(format!("the record does not validate: {}", errors.join("; "))) }
    }
}
