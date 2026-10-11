//! VER Records in the engine (`docs/ver.md`): LightCraft produces and consumes Visual Embedding Records to the
//! specification, through the `lightcraft-ver` core (CPNP-1, the identity hashes, the record builder, the schema and
//! profile validators).
//!
//! - **Producing** ([`produce`], `ver.record`, `ver.export`, the export hook in [`export`]): the photo's bytes are
//!   canonicalised (CPNP-1), hashed (`content_hash`, `pixel_hash`, PDQ, pHash), their metadata segments carried
//!   byte-exact (L1), reconciled into an IPTC view (L2, [`normalize`]), given an acquisition, a custody chain of
//!   registered actions and an honest C2PA status (L3), and embedded from the Canonical Buffer by an
//!   [`source::EmbeddingSource`]. The Record is validated (schema + profile) before it leaves: a Record with an
//!   error is never handed out ([`VerError::NotConformant`]).
//! - **Consuming** ([`consume::consume`], `ver.import`): a Record from anywhere is validated under its own schema and
//!   matched against the library by content, by rendered pixels (a grouping, errata E27), perceptually and, in
//!   LightCraft's own space, by vector.
//! - A camera raw (or anything else without a CPNP-1 path) has **no Record of its own**: no Canonical Buffer exists
//!   for it, so neither does a `pixel_hash`. Its exports do have Records, naming it as their source.

pub(crate) mod consume;
pub(crate) mod export;
pub(crate) mod normalize;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod sign;
pub(crate) mod source;
pub(crate) mod space;
#[cfg(test)]
mod tests;
pub(crate) mod util;

use std::sync::Arc;

use lightcraft_catalog::{PhotoId, Source};
use lightcraft_ver::cpnp::{CpnpError, Format};
use lightcraft_ver::hashing::Identity;
use lightcraft_ver::record::Builder;
pub use lightcraft_ver::record::{Level, Record, Schema};
use lightcraft_ver::segments::{Availability, RawSegment, State as Avail};
pub use lightcraft_ver::validate::Issue;
use lightcraft_ver::validate::Severity;
use serde_json::{Value, json};

use crate::Session;
use source::{EmbeddingSource, SpaceInfo};

/// `generator.name` of every Record LightCraft writes.
pub(crate) const GENERATOR: &str = "lightcraft";
/// The authority LightCraft's space ids and record-id namespace are minted under unless `ver.config {authority}`
/// says otherwise: the operator of this fork controls `verstandard.org` (VER 1.1 draft §11.6: an authority MUST be a
/// domain the publisher controls).
pub(crate) const DEFAULT_AUTHORITY: &str = "org.verstandard";
/// The six actions every VER 1.0 Record may carry (the 1.0.1 schema's closed enum).
const CORE_ACTIONS: [&str; 6] = ["decode", "normalize", "embed", "transcode", "redact", "import"];

/// Why no Record was produced (or accepted).
#[derive(Clone, Debug)]
pub enum VerError {
    /// The photo has no file of its own (a generated demo photo).
    NoAsset(String),
    /// The file has no CPNP-1 path (a camera raw, an unknown format, an unusable ICC profile): no Canonical Buffer, so
    /// no Record of its own.
    NoCpnpPath(String),
    /// The embedding model is not installed (or cannot run in this build); the sentence says how to get it.
    NeedsModel(String),
    /// The Record failed its schema or profile; it was not handed out.
    NotConformant(Vec<Issue>),
    /// The photo's bytes could not be read or decoded.
    Source(String),
    /// The model failed.
    Model(String),
    /// Bad input (an unknown photo, a document that is not a Record, a bad setting).
    Invalid(String),
}

impl VerError {
    /// A stable code for front ends and agents.
    pub fn code(&self) -> &'static str {
        match self {
            VerError::NoAsset(_) => "no_asset",
            VerError::NoCpnpPath(_) => "no_cpnp_path",
            VerError::NeedsModel(_) => "needs_model",
            VerError::NotConformant(_) => "not_conformant",
            VerError::Source(_) => "source",
            VerError::Model(_) => "model",
            VerError::Invalid(_) => "invalid",
        }
    }
}

impl std::fmt::Display for VerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerError::NoAsset(m) | VerError::Source(m) | VerError::Model(m) | VerError::Invalid(m) => write!(f, "{m}"),
            VerError::NoCpnpPath(m) => write!(f, "no VER Record: {m}"),
            VerError::NeedsModel(m) => write!(f, "a VER Record needs its canonical visual embedding: {m}"),
            VerError::NotConformant(issues) => {
                let errors: Vec<String> = issues
                    .iter()
                    .filter(|i| i.severity == Severity::Error)
                    .take(5)
                    .map(|i| format!("{} {}: {}", i.code, i.pointer, i.message))
                    .collect();
                write!(f, "the Record does not conform and was not handed out: {}", errors.join("; "))
            }
        }
    }
}

/// `ver.*` settings, saved with the library (`prefs.json` → `ver`).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub(crate) struct Prefs {
    /// The reverse-DNS authority space ids are minted under (default [`DEFAULT_AUTHORITY`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authority: Option<String>,
    /// Whether exports write Records beside the files they write (default on).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub export_sidecars: Option<bool>,
    /// An Ed25519 PKCS#8 PEM to sign Records with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sign_key_path: Option<String>,
    /// `1.1-draft` (default) or `1.0.1`: the schema `ver.record` uses when the call names none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_schema: Option<String>,
    /// This library's producer identity (`did:ver:local:<uuid>`), minted on first use: it scopes record ids (E16).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer_id: Option<String>,
}

impl Prefs {
    pub fn is_default(&self) -> bool {
        *self == Prefs::default()
    }
    pub fn authority(&self) -> String {
        self.authority.clone().unwrap_or_else(|| DEFAULT_AUTHORITY.to_string())
    }
    pub fn export_sidecars(&self) -> bool {
        self.export_sidecars.unwrap_or(true)
    }
    pub fn default_schema(&self) -> Schema {
        match self.default_schema.as_deref() {
            Some("1.0.1") => Schema::V1_0_1,
            _ => Schema::V1_1Draft,
        }
    }
}

/// The schema token of the commands (`1.0.1`, `1.1-draft`).
pub(crate) fn schema_name(s: Schema) -> &'static str {
    match s {
        Schema::V1_0_1 => "1.0.1",
        Schema::V1_1Draft => "1.1-draft",
    }
}

pub(crate) fn parse_schema(s: &str) -> Option<Schema> {
    match s {
        "1.0.1" | "1.0" => Some(Schema::V1_0_1),
        "1.1-draft" | "1.1" => Some(Schema::V1_1Draft),
        _ => None,
    }
}

pub(crate) fn parse_level(s: &str) -> Option<Level> {
    match s {
        "L0" => Some(Level::L0),
        "L1" => Some(Level::L1),
        "L2" => Some(Level::L2),
        "L3" => Some(Level::L3),
        _ => None,
    }
}

/// VER's session state ([`Session::ver`]).
#[derive(Default)]
pub(crate) struct State {
    pub prefs: Prefs,
    /// The last Record handed out: `{recordId, photo?, schema, level, at}`.
    pub last_produced: Option<Value>,
    /// The weights digest of the model file at this path, computed once ([`space::weights_sha256`]).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub weights: Option<(std::path::PathBuf, Arc<std::sync::OnceLock<Result<String, String>>>)>,
    /// The model, loaded for VER while the vector lane is switched off (the lane's own copy is used otherwise).
    #[cfg(not(target_arch = "wasm32"))]
    pub model: Option<(std::path::PathBuf, Arc<lightcraft_embed::Model>)>,
    /// Canonical vectors of Records produced this session, by content key, with their space id (the consumer compares
    /// against them too).
    pub vectors: std::collections::HashMap<String, (String, Vec<f32>)>,
    /// Background `ver.export` work ([`export::pump`]).
    pub queue: export::Queue,
    /// A test's embedding source, used instead of the model.
    #[cfg(test)]
    pub test_source: Option<Arc<dyn EmbeddingSource>>,
}

impl State {
    /// Another library was opened: its settings are read afresh; the loaded model stays.
    pub fn library_changed(&mut self, prefs: Prefs) {
        self.prefs = prefs;
        self.last_produced = None;
        self.vectors.clear();
        self.queue = export::Queue::default();
    }
}

/// What producing needs to know about the producer: who it is, under which authority, and how it signs.
#[derive(Clone)]
pub(crate) struct Context {
    pub authority: String,
    pub producer_id: String,
    #[cfg(not(target_arch = "wasm32"))]
    pub signer: Option<sign::Signer>,
}

/// This library's producer id, minted (and saved) on first use.
fn producer_id(s: &mut Session) -> String {
    if let Some(p) = s.ver.prefs.producer_id.clone().filter(|p| p.starts_with("did:")) {
        return p;
    }
    let salt = s.library_folder.clone().unwrap_or_default();
    let minted = format!("did:ver:local:{}", util::uuid4(&salt));
    s.ver.prefs.producer_id = Some(minted.clone());
    if let Err(e) = s.save_prefs() {
        log::warn!("ver: could not save the producer id: {e}");
    }
    minted
}

/// The producer context of this session: an error when a configured signing key cannot be read (a Record the user
/// asked to have signed is not written unsigned).
pub(crate) fn context(s: &mut Session) -> Result<Context, VerError> {
    let producer_id = producer_id(s);
    #[cfg(not(target_arch = "wasm32"))]
    let signer = match s.ver.prefs.sign_key_path.clone() {
        Some(p) => Some(sign::Signer::load(std::path::Path::new(&p)).map_err(VerError::Invalid)?),
        None => None,
    };
    Ok(Context {
        authority: s.ver.prefs.authority(),
        producer_id,
        #[cfg(not(target_arch = "wasm32"))]
        signer,
    })
}

/// The embedding source of this session: the vector lane's model when it is installed, else one that says why not.
pub(crate) fn embedding_source(s: &mut Session) -> Arc<dyn EmbeddingSource> {
    #[cfg(test)]
    if let Some(t) = &s.ver.test_source {
        return t.clone();
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        Arc::new(source::Unavailable(VerError::NeedsModel(crate::vera::lane::LaneState::NeedsDesktop.reason())))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::vera::lane::{self, LaneState};
        let model = match lane::state(s) {
            LaneState::Live | LaneState::Error(_) => lane::model(s).map_err(VerError::Model),
            LaneState::DisabledByUser => own_model(s),
            st => Err(VerError::NeedsModel(st.reason())),
        };
        let model = match model {
            Ok(m) => m,
            Err(e) => return Arc::new(source::Unavailable(e)),
        };
        let Some(dir) = s.vera_model_dir.clone() else {
            return Arc::new(source::Unavailable(VerError::NeedsModel(LaneState::NeedsModel(None).reason())));
        };
        let path = dir.join(lightcraft_embed::WEIGHTS_FILE);
        let weights = match &s.ver.weights {
            Some((p, w)) if *p == path => w.clone(),
            _ => {
                let w = Arc::new(std::sync::OnceLock::new());
                s.ver.weights = Some((path.clone(), w.clone()));
                w
            }
        };
        Arc::new(source::LaneSource { model, authority: s.ver.prefs.authority(), weights, weights_path: path })
    }
}

/// Whether Records can be produced, cheaply: the space id when the model is installed (nothing is loaded or hashed),
/// else why not. The weights digest is reported when it is already known.
pub(crate) fn readiness(s: &mut Session) -> Result<(String, Option<String>), VerError> {
    #[cfg(test)]
    if let Some(t) = &s.ver.test_source {
        return t.space().map(|sp| (sp.space_id, sp.descriptor.pointer("/model/weights_sha256").and_then(Value::as_str).map(str::to_string)));
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = s;
        Err(VerError::NeedsModel(crate::vera::lane::LaneState::NeedsDesktop.reason()))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::vera::lane::{self, LaneState};
        match lane::state(s) {
            LaneState::Live | LaneState::Error(_) | LaneState::DisabledByUser => {
                let weights = s.ver.weights.as_ref().and_then(|(_, w)| w.get().cloned()).and_then(Result::ok);
                Ok((space::space_id(&s.ver.prefs.authority()), weights))
            }
            st => Err(VerError::NeedsModel(st.reason())),
        }
    }
}

/// The model loaded for VER alone (the vector lane is switched off for this library).
#[cfg(not(target_arch = "wasm32"))]
fn own_model(s: &mut Session) -> Result<Arc<lightcraft_embed::Model>, VerError> {
    let dir = s.vera_model_dir.clone().ok_or_else(|| VerError::NeedsModel(crate::vera::lane::LaneState::NeedsModel(None).reason()))?;
    if let Some((d, m)) = &s.ver.model
        && *d == dir
    {
        return Ok(m.clone());
    }
    let m = Arc::new(lightcraft_embed::Model::load(&dir, lightcraft_embed::best_device()).map_err(|e| VerError::Model(e.to_string()))?);
    s.ver.model = Some((dir, m.clone()));
    Ok(m)
}

/// Everything learned from an Asset's bytes: the heavy part of producing a Record, done once for both schemas of a
/// dual publication, and safe to do off the session thread.
#[derive(Clone, Debug)]
pub(crate) struct Analysis {
    pub format: Format,
    pub animated: bool,
    pub alpha_present: bool,
    pub hdr_present: bool,
    pub identity: Identity,
    pub byte_length: u64,
    pub availability: Availability,
    pub segments: Vec<RawSegment>,
    pub space: SpaceInfo,
    pub vector: Vec<f32>,
    pub decoded_at: String,
    pub normalized_at: String,
    pub embedded_at: String,
    /// PDQ's quality figure of the Canonical Buffer (for the digest lane's fingerprint).
    pub pdq_quality: Option<u8>,
    /// The in-asset descriptive metadata, per family (for the normalized view).
    pub in_asset: Arc<normalize::InAsset>,
}

/// Why `bytes` have no CPNP-1 path, or `None` when they have one ([`Format`]).
pub(crate) fn cpnp_format(bytes: &[u8]) -> Result<Format, VerError> {
    // a raw is TIFF-shaped more often than not; it is a raw first
    if let Some(raw) = lightcraft_raw::probe(bytes) {
        return Err(VerError::NoCpnpPath(format!(
            "a camera raw ({raw:?}) has no CPNP-1 path, so it has no Canonical Buffer and no Record of its own; its exports have Records that name it"
        )));
    }
    lightcraft_ver::cpnp::is_cpnp_format(bytes).ok_or_else(|| {
        VerError::NoCpnpPath(
            "this file format has no CPNP-1 path (VER 1.0 §4 covers JPEG, PNG, WebP, AVIF, HEIC, TIFF, BMP and GIF), so it has no Record of its own"
                .into(),
        )
    })
}

fn cpnp_error(e: CpnpError) -> VerError {
    match e {
        CpnpError::Undecodable(m) => VerError::Source(format!("the file could not be decoded: {m}")),
        CpnpError::UnusableIcc(m) => VerError::NoCpnpPath(format!("its embedded ICC profile is unusable ({m}); CPNP-1 fails closed (errata E8)")),
        CpnpError::Unsupported(m) => VerError::NoCpnpPath(m),
    }
}

/// Canonicalise, hash, read the segments of and embed `bytes`.
pub(crate) fn analyze(bytes: &[u8], embed: &dyn EmbeddingSource) -> Result<Analysis, VerError> {
    let format = cpnp_format(bytes)?;
    // the model's absence is known before anything is decoded
    let space = embed.space()?;
    let decoded_at = util::now();
    let (buf, identity, pdq_quality) = lightcraft_ver::hashing::identity_with_quality(bytes).map_err(cpnp_error)?;
    let pdq_quality = Some(pdq_quality);
    let normalized_at = util::now();
    let (availability, segments) = lightcraft_ver::segments::segments(bytes, format);
    let vector = embed.embed(&buf)?;
    if vector.len() != space.dim {
        return Err(VerError::Model(format!("the embedding has {} numbers, its space {}", vector.len(), space.dim)));
    }
    Ok(Analysis {
        format,
        animated: buf.animated,
        alpha_present: buf.alpha_present,
        hdr_present: buf.hdr_present,
        identity,
        byte_length: bytes.len() as u64,
        availability,
        segments,
        space,
        vector,
        decoded_at,
        normalized_at,
        embedded_at: util::now(),
        pdq_quality,
        in_asset: Arc::new(normalize::InAsset::read(bytes)),
    })
}

/// A custody event before decoding (import, develop, export).
#[derive(Clone, Debug)]
pub(crate) struct Event {
    pub action: &'static str,
    pub at: String,
    /// Members only VER 1.1 has (`uri`, `target`); dropped from a 1.0 Record.
    pub extra_1_1: Value,
}

/// What the catalog (or the export) knows about an Asset beyond its bytes.
#[derive(Clone, Debug, Default)]
pub(crate) struct Facts {
    /// `image.source_bytes_ref.uri` (a `file://` URI), when the Asset has a location worth naming.
    pub uri: Option<String>,
    /// `provenance.acquisition`: method, `acquired_at`, extra members (`source_uri`, `actor`).
    pub acquisition: Option<(String, String, Value)>,
    pub events: Vec<Event>,
    /// The catalog's descriptive fields (platform values, errata E14); `None` to assert only what the file says.
    pub platform: Option<normalize::Platform>,
    /// A `lineage.parents[]` entry naming the Asset this one was made from (VER 1.1 §3).
    pub lineage: Option<Value>,
    /// A source with no Record of its own (a raw), carried in the 1.1 `extensions` block.
    pub source_note: Option<Value>,
}

/// One Record, as handed out.
#[derive(Clone, Debug)]
pub struct Produced {
    pub record: Record,
    pub issues: Vec<Issue>,
    pub level: Level,
    /// Why the Record stops below the level asked for, and anything else worth saying.
    pub notes: Vec<String>,
}

impl Produced {
    pub fn to_json(&self) -> Value {
        json!({
            "record": self.record.0,
            "issues": self.issues,
            "conformant": lightcraft_ver::validate::is_conformant(&self.issues),
            "level": self.level.as_str(),
            "notes": self.notes,
        })
    }
}

/// The C2PA status a Record may honestly state, with the `reason` 1.1 requires for `unverifiable`: `None` when no
/// honest status exists for this schema (the Record then stops below L3, errata E28).
fn c2pa_status(a: &Availability, schema: Schema, authority: &str, notes: &mut Vec<String>) -> Option<(&'static str, Option<String>)> {
    match a.c2pa {
        Avail::Absent => Some(("missing", None)),
        Avail::Present if schema == Schema::V1_1Draft => {
            notes.push("The file carries a C2PA manifest; LightCraft does not validate manifests, so its status is `unverifiable` and nothing from it is trusted.".into());
            Some(("unverifiable", Some(format!("{authority}/x-validation_not_implemented"))))
        }
        Avail::Present => {
            notes.push("The file carries a C2PA manifest LightCraft cannot validate, and VER 1.0 has no `unverifiable` status: rather than state a status contrary to fact, this 1.0 Record omits provenance.c2pa and stops at L2 (errata E28).".into());
            None
        }
        _ => {
            notes.push("Whether the file carries a C2PA manifest is unknown, so no C2PA status is stated and the Record stops at L2.".into());
            None
        }
    }
}

/// Assemble, sign and validate one Record from an [`Analysis`] and the [`Facts`] around it.
pub(crate) fn assemble(ctx: &Context, a: &Analysis, f: &Facts, schema: Schema, target: Level) -> Result<Produced, VerError> {
    let now = util::now();
    let mut notes = Vec::new();
    let is_1_1 = schema == Schema::V1_1Draft;
    let normalized = normalize::reconcile(&a.in_asset, f.platform.as_ref());
    let c2pa = c2pa_status(&a.availability, schema, &ctx.authority, &mut notes);
    let mut level = Level::L1;
    if normalized.view.is_some() {
        level = Level::L2;
        if c2pa.is_some() && f.acquisition.is_some() {
            level = Level::L3;
        }
    } else {
        notes
            .push("L2 needs at least one descriptive field (a title, caption, keywords, creator…); neither the file nor the catalog has one.".into());
    }
    if target < level {
        level = target;
    } else if target > level {
        notes.push(format!("Asked for {}, the evidence supports {}.", target.as_str(), level.as_str()));
    }
    let domain = space::domain(&ctx.authority);
    let record_id = util::record_id(&domain, &ctx.producer_id, &a.identity.content_hash, schema.ver_version());
    let version = env!("CARGO_PKG_VERSION");
    let tool = format!("{GENERATOR}/{version}");
    let mut b = Builder::new(schema, GENERATOR, version)
        .record_id(&record_id)
        .created_at(&now)
        .identity(&a.identity, a.format, a.animated, a.alpha_present, a.hdr_present)
        .availability(&a.availability)
        .raw_segments(&a.segments)
        .space(a.space.descriptor.clone())
        .embedding("canonical_visual", &a.space.space_id, a.space.dim, &a.space.dtype, &a.vector, &a.embedded_at, &a.space.producer);
    if let Some(uri) = &f.uri {
        b = b.source_bytes_ref(uri, &a.identity.content_hash, a.byte_length);
    }
    if let Some(view) = &normalized.view {
        b = b.normalized(view.clone(), normalized.trust.clone());
    }
    if let Some((method, at, extra)) = &f.acquisition {
        b = b.acquisition(method, at, extra.clone());
    }
    for e in &f.events {
        if is_1_1 {
            b = b.chain_event(e.action, GENERATOR, &tool, &e.at, e.extra_1_1.clone());
        } else if CORE_ACTIONS.contains(&e.action) {
            b = b.chain_event(e.action, GENERATOR, &tool, &e.at, json!({}));
        }
    }
    b = b
        .chain_event("decode", GENERATOR, &format!("{tool} (lightcraft-ver CPNP-1 decode)"), &a.decoded_at, json!({}))
        .chain_event("normalize", GENERATOR, &format!("{tool} (lightcraft-ver CPNP-1)"), &a.normalized_at, json!({}))
        .chain_event("embed", GENERATOR, &a.space.producer, &a.embedded_at, json!({}));
    // The builder refuses a document with an error at its claimed level; what it has no setter for (the level, the C2PA
    // block with its 1.1 `reason`, lineage, compute, extensions, conflicts, the signature) is added to the document it
    // builds, and the whole is validated again below before it is handed out.
    let mut rec = b.build().map_err(|e| VerError::Invalid(format!("the Record could not be built: {e}")))?;
    let doc = &mut rec.0;
    doc["conformance_level"] = json!(level.as_str());
    if let Some((status, reason)) = &c2pa
        && let Some(p) = doc.get_mut("provenance").and_then(Value::as_object_mut)
    {
        let mut c = json!({"status": status});
        if is_1_1 && let Some(r) = reason {
            c["reason"] = json!(r);
            c["error_detail"] = json!("LightCraft does not validate C2PA manifests.");
        }
        p.insert("c2pa".into(), c);
    }
    if !normalized.conflicts.is_empty()
        && let Some(m) = doc.get_mut("metadata").and_then(Value::as_object_mut)
    {
        m.insert("conflicts".into(), json!(normalized.conflicts));
    }
    if is_1_1 {
        if let Some(e) = doc.pointer_mut("/embeddings/0").and_then(Value::as_object_mut) {
            e.insert("compute".into(), a.space.compute.clone());
        }
        if let Some(parent) = &f.lineage {
            doc["lineage"] = json!({"parents": [parent]});
        }
        let mut ext = serde_json::Map::new();
        if !normalized.origins.is_empty() {
            ext.insert("field_origins".into(), Value::Object(normalized.origins.clone()));
        }
        if let Some(src) = &f.source_note {
            ext.insert("source".into(), src.clone());
        }
        ext.insert("producer_id".into(), json!(ctx.producer_id));
        doc["extensions"] = json!({ format!("{}.lightcraft", ctx.authority): ext });
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(signer) = &ctx.signer {
        if is_1_1 && let Some(chain) = doc.pointer_mut("/provenance/chain").and_then(Value::as_array_mut) {
            chain.push(json!({"at": now, "actor": GENERATOR, "action": "sign", "target": "record", "tool": tool}));
        }
        signer.sign(doc).map_err(VerError::Invalid)?;
    }
    let issues = lightcraft_ver::validate::all(&rec.0, schema);
    if !lightcraft_ver::validate::is_conformant(&issues) {
        return Err(VerError::NotConformant(issues));
    }
    Ok(Produced { record: rec, issues, level, notes })
}

/// How a Record is asked for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Opts {
    pub schema: Schema,
    pub level: Level,
}

/// What the catalog knows about library photo `p` (whose file is at `path`), for a Record of the photo itself.
pub(crate) fn photo_facts(s: &Session, p: &lightcraft_catalog::Photo, path: &str) -> Facts {
    let now = util::now();
    let acquired = util::catalog_time(&p.imported, &now).unwrap_or_else(|| now.clone());
    let uri = util::file_uri(std::path::Path::new(path));
    Facts {
        uri: Some(uri.clone()),
        acquisition: Some(("user_upload".into(), acquired.clone(), json!({"source_uri": uri, "actor": GENERATOR}))),
        events: vec![Event { action: "import", at: acquired, extra_1_1: json!({}) }],
        platform: Some(normalize::Platform::of(p, &s.catalog)),
        lineage: None,
        source_note: None,
    }
}

/// The bytes of photo `id`'s file, what the catalog knows about it and its content key.
pub(crate) fn photo_asset(s: &mut Session, id: PhotoId) -> Result<(Vec<u8>, Facts, String), VerError> {
    let p = s.catalog.photo(id).cloned().ok_or_else(|| VerError::Invalid(format!("no photo {}", id.0)))?;
    let Source::File { path } = &p.source else {
        return Err(VerError::NoAsset(format!("{} is a generated demo photo: it has no file, so it has no Record", p.file_name)));
    };
    let bytes = match &s.media.file_bytes {
        Some(read) => read(path),
        None => std::fs::read(path).map_err(|e| format!("{path}: {e}")),
    }
    .map_err(VerError::Source)?;
    Ok((bytes, photo_facts(s, &p, path), crate::media::content_key(&p)))
}

/// Remember what producing learned about a library photo: its Canonical-Buffer fingerprint for the digest lane and
/// its vector for the consumer.
pub(crate) fn remember(s: &mut Session, key: &str, a: &Analysis) {
    crate::vera::ensure_open(s);
    if let Some(fp) = crate::vera::digests::Fingerprint::from_identity(&a.identity, a.pdq_quality) {
        s.vera.digests.insert_fingerprint(key.to_string(), fp);
    }
    s.ver.vectors.insert(key.to_string(), (a.space.space_id.clone(), a.vector.clone()));
}

/// A Record of photo `id`, validated, or why there is none.
pub(crate) fn produce(s: &mut Session, id: PhotoId, opts: Opts) -> Result<Produced, VerError> {
    let source = embedding_source(s);
    let (bytes, facts, key) = photo_asset(s, id)?;
    let ctx = context(s)?;
    let a = analyze(&bytes, source.as_ref())?;
    remember(s, &key, &a);
    let out = assemble(&ctx, &a, &facts, opts.schema, opts.level)?;
    note_produced(s, &out, Some(id), opts.schema);
    Ok(out)
}

/// The two Records of a dual publication (VER 1.1 §1.1): a 1.0 Record and a 1.1 Record of the same bytes, sharing
/// `content_hash`.
pub(crate) fn produce_both(ctx: &Context, a: &Analysis, f: &Facts, level: Level) -> Result<(Produced, Produced), VerError> {
    Ok((assemble(ctx, a, f, Schema::V1_0_1, level)?, assemble(ctx, a, f, Schema::V1_1Draft, level)?))
}

pub(crate) fn note_produced(s: &mut Session, p: &Produced, photo: Option<PhotoId>, schema: Schema) {
    s.ver.last_produced = Some(json!({
        "recordId": p.record.0.get("record_id"),
        "photo": photo.map(|i| i.0),
        "schema": schema_name(schema),
        "level": p.level.as_str(),
        "at": p.record.0.get("created_at"),
    }));
}
