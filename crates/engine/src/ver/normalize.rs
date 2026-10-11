//! `metadata.normalized` (L2): the reconciled view of a picture's descriptive metadata, in the IPTC Photo Metadata
//! vocabulary as XMP spells it (`dc:title`, `dc:description`, `Iptc4xmpCore:AltTextAccessibility`, …), with one
//! `metadata.trust` entry per assertion and a `metadata.conflicts` entry wherever two sources disagree.
//!
//! Reconciliation follows errata E14: **XMP > IPTC-IIM > EXIF > platform**, where *platform* is everything the
//! catalog knows that the file does not say (a title typed in LightCraft, a caption accepted from Describe). A platform
//! value fills a field the file leaves empty, never wins against the file, and is recorded in `conflicts` with
//! `source: "platform"` when it disagrees. Every tier is `unsigned`: the schema's tiers grade *cryptographic* evidence
//! (`c2pa_verified`, `signed_unverified`), LightCraft verifies no C2PA manifest, and a value typed by the user or
//! accepted from a model is a platform assertion (E14: "always carries trust tier `unsigned`"). Where each value came
//! from inside LightCraft (typed, accepted from Describe, read from the file) is carried, in 1.1 Records, by the
//! `extensions` block ([`Normalized::origins`]).

use std::collections::BTreeMap;

use lightcraft_meta::Metadata;
use serde_json::{Map, Value, json};

/// The JSON-LD prefixes the view may use.
const CONTEXT: [(&str, &str); 3] = [
    ("dc", "http://purl.org/dc/elements/1.1/"),
    ("Iptc4xmpCore", "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"),
    ("photoshop", "http://ns.adobe.com/photoshop/1.0/"),
];

/// The fields, in the order the view lists them.
pub(crate) const FIELDS: [&str; 12] = [
    "dc:title",
    "dc:description",
    "Iptc4xmpCore:AltTextAccessibility",
    "Iptc4xmpCore:ExtDescrAccessibility",
    "dc:subject",
    "dc:creator",
    "dc:rights",
    "photoshop:DateCreated",
    "Iptc4xmpCore:Location",
    "photoshop:City",
    "photoshop:State",
    "photoshop:Country",
];

fn text(s: Option<&str>) -> Option<Value> {
    s.map(str::trim).filter(|t| !t.is_empty()).map(|t| json!(t))
}

fn list<'a>(items: impl Iterator<Item = &'a String>) -> Option<Value> {
    let v: Vec<&str> = items.map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    (!v.is_empty()).then(|| json!(v))
}

/// The view's value of `field` in one source's metadata.
fn field_of(m: &Metadata, field: &str) -> Option<Value> {
    match field {
        "dc:title" => text(m.title.as_deref()),
        "dc:description" => text(m.caption.as_deref()),
        "Iptc4xmpCore:AltTextAccessibility" => text(m.alt_text.as_deref()),
        "Iptc4xmpCore:ExtDescrAccessibility" => text(m.extended_description.as_deref()),
        "dc:subject" => list(m.keywords.iter()),
        "dc:creator" => text(m.artist.as_deref()).map(|a| json!([a])),
        "dc:rights" => text(m.copyright.as_deref()),
        "photoshop:DateCreated" => m.capture_time.as_ref().map(|t| json!(t.to_iso())),
        "Iptc4xmpCore:Location" => text(m.sublocation.as_deref()),
        "photoshop:City" => text(m.city.as_deref()),
        "photoshop:State" => text(m.state.as_deref()),
        "photoshop:Country" => text(m.country.as_deref()),
        _ => None,
    }
}

/// What the catalog says about a photo, per view field, and where LightCraft got each value (`user`, `describe`, `file`
/// or `catalog`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Platform {
    pub fields: BTreeMap<&'static str, (Value, String)>,
}

impl Platform {
    /// The catalog's descriptive fields of `p` (keywords as an export writes them).
    pub fn of(p: &lightcraft_catalog::Photo, catalog: &lightcraft_catalog::Catalog) -> Platform {
        use lightcraft_catalog::Provenance;
        let m = &p.meta;
        let origin = |key: &str| match m.provenance.get(key) {
            Some(Provenance::User) => "user".to_string(),
            Some(Provenance::Exif) => "file".to_string(),
            Some(Provenance::Describe { model, .. }) => format!("describe:{model}"),
            None => "catalog".to_string(),
        };
        let mut fields = BTreeMap::new();
        let mut put = |k: &'static str, v: Option<Value>, o: String| {
            if let Some(v) = v {
                fields.insert(k, (v, o));
            }
        };
        put("dc:title", text(Some(&m.title)), origin("title"));
        put("dc:description", text(Some(&m.caption)), origin("caption"));
        put("Iptc4xmpCore:AltTextAccessibility", text(Some(&m.alt_text)), origin("alt_text"));
        put("Iptc4xmpCore:ExtDescrAccessibility", text(Some(&m.extended_description)), origin("extended_description"));
        put("dc:subject", list(catalog.export_keywords(&m.keywords).flat.iter()), origin("keywords"));
        put("dc:creator", text(Some(&m.creator)).map(|a| json!([a])), "catalog".into());
        put("dc:rights", text(Some(&m.copyright)), "catalog".into());
        put("photoshop:DateCreated", text(p.captured.as_deref()), "catalog".into());
        put("Iptc4xmpCore:Location", text(Some(&m.location)), "catalog".into());
        put("photoshop:City", text(Some(&m.city)), "catalog".into());
        put("photoshop:State", text(Some(&m.state)), "catalog".into());
        put("photoshop:Country", text(Some(&m.country)), "catalog".into());
        Platform { fields }
    }
}

/// The in-asset metadata of a file, one parse per family (E14's sources).
#[derive(Debug)]
pub(crate) struct InAsset {
    pub sources: Vec<(&'static str, Metadata)>,
}

impl InAsset {
    /// Read the XMP, IPTC-IIM and EXIF blocks of `bytes` (in that precedence).
    pub fn read(bytes: &[u8]) -> InAsset {
        let e = lightcraft_meta::embedded(bytes);
        let mut sources = Vec::new();
        if let Some(x) = e.xmp.as_deref().and_then(|x| lightcraft_meta::parse_xmp(x).ok()) {
            sources.push(("xmp", x.metadata));
        }
        if let Some(i) = e.iptc.as_deref() {
            sources.push(("iptc-iim", lightcraft_meta::parse_iptc(i)));
        }
        if let Some(x) = e.exif.as_deref() {
            sources.push(("exif", lightcraft_meta::read_exif(x)));
        }
        InAsset { sources }
    }
}

/// The reconciled view.
#[derive(Clone, Debug, Default)]
pub(crate) struct Normalized {
    /// `metadata.normalized` (with its `@context`), or `None` when nothing is asserted.
    pub view: Option<Value>,
    /// `metadata.trust`: `(field, tier)` per assertion.
    pub trust: Vec<(String, String)>,
    /// `metadata.conflicts`.
    pub conflicts: Vec<Value>,
    /// Where each asserted value came from (`xmp`, `iptc-iim`, `exif`, or `platform:<user|describe:model|catalog|file>`).
    pub origins: Map<String, Value>,
}

/// Two values the same assertion? Strings compare trimmed; lists as sets.
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            fn set(v: &[Value]) -> std::collections::BTreeSet<&str> {
                v.iter().filter_map(Value::as_str).map(str::trim).collect()
            }
            set(x) == set(y)
        }
        (Value::String(x), Value::String(y)) => x.trim() == y.trim(),
        _ => a == b,
    }
}

/// Reconcile the file's metadata with the catalog's (`platform`, `None` to assert only what the file says).
pub(crate) fn reconcile(asset: &InAsset, platform: Option<&Platform>) -> Normalized {
    let mut view = Map::new();
    let mut out = Normalized::default();
    for field in FIELDS {
        // every distinct value, in precedence order, with its source
        let mut values: Vec<(String, Value)> = Vec::new();
        for (source, m) in &asset.sources {
            if let Some(v) = field_of(m, field) {
                values.push(((*source).to_string(), v));
            }
        }
        let platform_value = platform.and_then(|p| p.fields.get(field));
        if let Some((v, _)) = platform_value {
            values.push(("platform".into(), v.clone()));
        }
        let Some((winner_source, winner)) = values.first().cloned() else { continue };
        view.insert(field.to_string(), winner.clone());
        out.trust.push((field.to_string(), "unsigned".into()));
        let origin = match (winner_source.as_str(), platform_value) {
            ("platform", Some((_, o))) => format!("platform:{o}"),
            (s, _) => s.to_string(),
        };
        out.origins.insert(field.to_string(), json!(origin));
        // one entry per source that disagrees with another (sources are distinct by construction)
        if values.iter().any(|(_, v)| !same(v, &winner)) {
            let entries: Vec<Value> = values.iter().map(|(s, v)| json!({"source": s, "value": v, "trust": "unsigned"})).collect();
            out.conflicts.push(json!({"field": field, "values": entries}));
        }
    }
    if !view.is_empty() {
        let used: Vec<&str> = view.keys().filter_map(|k| k.split_once(':').map(|(p, _)| p)).collect();
        let ctx: Map<String, Value> = CONTEXT.iter().filter(|(p, _)| used.contains(p)).map(|(p, u)| (p.to_string(), json!(u))).collect();
        let mut full = Map::new();
        full.insert("@context".into(), Value::Object(ctx));
        full.extend(view);
        out.view = Some(Value::Object(full));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(title: &str, keywords: &[&str]) -> Metadata {
        Metadata { title: Some(title.into()), keywords: keywords.iter().map(|s| s.to_string()).collect(), ..Default::default() }
    }

    #[test]
    fn file_wins_platform_fills_and_conflicts_are_recorded() {
        let asset = InAsset {
            sources: vec![("xmp", meta("Harbour", &["sea", "boats"])), ("exif", Metadata { artist: Some("Ann".into()), ..Default::default() })],
        };
        let mut platform = Platform::default();
        platform.fields.insert("dc:title", (json!("Harbour at dusk"), "user".into()));
        platform.fields.insert("dc:subject", (json!(["boats", "sea"]), "catalog".into()));
        platform.fields.insert("Iptc4xmpCore:AltTextAccessibility", (json!("Boats moored in a harbour"), "describe:llava".into()));
        let n = reconcile(&asset, Some(&platform));
        let v = n.view.unwrap();
        assert_eq!(v["dc:title"], "Harbour", "the file wins (E14)");
        assert_eq!(v["Iptc4xmpCore:AltTextAccessibility"], "Boats moored in a harbour", "platform fills an empty field");
        assert_eq!(v["dc:creator"], json!(["Ann"]));
        assert_eq!(v["@context"]["dc"], "http://purl.org/dc/elements/1.1/");
        assert!(v["@context"].get("photoshop").is_none(), "only the prefixes used");
        // trust per assertion, all unsigned
        assert_eq!(n.trust.len(), 4);
        assert!(n.trust.iter().all(|(_, t)| t == "unsigned"));
        // the title conflict names both sources; the keyword sets agree, so no conflict there
        assert_eq!(n.conflicts.len(), 1, "{:?}", n.conflicts);
        assert_eq!(n.conflicts[0]["field"], "dc:title");
        assert_eq!(n.conflicts[0]["values"][1]["source"], "platform");
        assert_eq!(n.origins["Iptc4xmpCore:AltTextAccessibility"], "platform:describe:llava");
        assert_eq!(n.origins["dc:title"], "xmp");
        // nothing anywhere: no view
        assert!(reconcile(&InAsset { sources: vec![] }, None).view.is_none());
    }
}
