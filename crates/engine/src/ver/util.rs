//! Small pieces a VER Record needs that no dependency provides: base64 (RFC 4648 §4 and §5), the RFC 8785 JSON
//! Canonicalization Scheme, UUIDv5 record ids and RFC 3339 timestamps.

// the web build signs nothing: JCS and base64url serve the signer
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use serde_json::Value;
use sha1::Digest as _;
use sha2::Sha256;

const STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn encode_with(bytes: &[u8], alphabet: &[u8; 64], pad: bool) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let sym = |i: u32| char::from(alphabet.get((i & 63) as usize).copied().unwrap_or(b'A'));
    for chunk in bytes.chunks(3) {
        let b = [chunk.first().copied().unwrap_or(0), chunk.get(1).copied().unwrap_or(0), chunk.get(2).copied().unwrap_or(0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(sym(n >> 18));
        out.push(sym(n >> 12));
        if chunk.len() > 1 {
            out.push(sym(n >> 6));
        } else if pad {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(sym(n));
        } else if pad {
            out.push('=');
        }
    }
    out
}

/// Standard base64, padded, unwrapped (RFC 4648 §4).
#[cfg(test)]
pub(crate) fn b64(bytes: &[u8]) -> String {
    encode_with(bytes, STD, true)
}

/// base64url without padding (RFC 4648 §5, as JWS uses it).
pub(crate) fn b64url(bytes: &[u8]) -> String {
    encode_with(bytes, URL, false)
}

/// Decode standard base64 (padding optional, no whitespace); `None` for anything else.
pub(crate) fn unb64(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        let v = STD.iter().position(|s| *s == c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    Some(out)
}

/// Lowercase hex.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase hex → bytes; `None` unless every character is a hex digit and there is an even number of them.
pub(crate) fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len()).step_by(2).map(|i| text.get(i..i + 2).and_then(|p| u8::from_str_radix(p, 16).ok())).collect()
}

/// SHA-256 as 64 lowercase hex.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// RFC 8785 (JCS) serialization of `v`: object members sorted by their UTF-16 code units, no insignificant
/// whitespace, strings escaped as ECMAScript's `JSON.stringify` does and numbers written as ECMAScript's
/// `Number.prototype.toString`. An error for a number JSON cannot carry (NaN, infinity).
pub(crate) fn jcs(v: &Value) -> Result<String, String> {
    let mut out = String::new();
    jcs_into(v, &mut out, 0)?;
    Ok(out)
}

fn jcs_into(v: &Value, out: &mut String, depth: usize) -> Result<(), String> {
    if depth > 128 {
        return Err("JSON nested too deeply to canonicalize".into());
    }
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&jcs_number(n)?),
        Value::String(s) => jcs_string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_into(x, out, depth + 1)?;
            }
            out.push(']');
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                jcs_string(k, out);
                out.push(':');
                if let Some(x) = o.get(k) {
                    jcs_into(x, out, depth + 1)?;
                }
            }
            out.push('}');
        }
    }
    Ok(())
}

fn jcs_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// ECMAScript `Number.prototype.toString` of a JSON number (RFC 8785 §3.2.2.3).
fn jcs_number(n: &serde_json::Number) -> Result<String, String> {
    let f = match (n.as_i64(), n.as_u64()) {
        // integers JSON can carry exactly in a double print as themselves
        (Some(i), _) if i.unsigned_abs() < (1u64 << 53) => return Ok(i.to_string()),
        (_, Some(u)) if u < (1u64 << 53) => return Ok(u.to_string()),
        _ => n.as_f64().ok_or("not a number")?,
    };
    if !f.is_finite() {
        return Err("NaN and infinity have no JSON form".into());
    }
    if f == 0.0 {
        return Ok("0".into());
    }
    let sign = if f < 0.0 { "-" } else { "" };
    // Rust's `{:e}` is the shortest round-tripping form, as ECMAScript's: "d.ddde±x"
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').ok_or("unexpected float form")?;
    let exp: i32 = exp.parse().map_err(|_| "unexpected float exponent")?;
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let k = digits.len() as i32;
    let n = exp + 1;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (a, b) = digits.split_at(n as usize);
        format!("{a}.{b}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let sign_e = if e < 0 { "-" } else { "+" };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() { format!("{first}e{sign_e}{}", e.abs()) } else { format!("{first}.{rest}e{sign_e}{}", e.abs()) }
    };
    Ok(format!("{sign}{body}"))
}

/// The RFC 4122 DNS namespace UUID.
const NAMESPACE_DNS: [u8; 16] = [0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8];

/// UUIDv5 (RFC 9562 §5.5) of `name` in `namespace`.
pub(crate) fn uuid5(namespace: &[u8; 16], name: &str) -> [u8; 16] {
    let mut h = sha1::Sha1::new();
    h.update(namespace);
    h.update(name.as_bytes());
    let d = h.finalize();
    let mut u = [0u8; 16];
    u.copy_from_slice(d.get(..16).unwrap_or(&[0; 16]));
    u[6] = (u[6] & 0x0f) | 0x50;
    u[8] = (u[8] & 0x3f) | 0x80;
    u
}

/// A UUID in its canonical text form.
pub(crate) fn uuid_text(u: &[u8; 16]) -> String {
    let h = hex(u);
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

/// The namespace record ids are minted in: UUIDv5 of the publisher's domain in the DNS namespace (for the default
/// authority `org.verstandard`, `uuid5(DNS, "verstandard.org")`, the namespace the reference producer uses).
pub(crate) fn record_namespace(domain: &str) -> [u8; 16] {
    uuid5(&NAMESPACE_DNS, domain)
}

/// A record id (errata E16): UUIDv5 over `producer | content_hash | ver_version`, so re-issuing a Record for the same
/// bytes by the same producer under the same version is idempotent, two producers never collide, and the 1.0 and 1.1
/// twins of a dual publication are two documents.
pub(crate) fn record_id(domain: &str, producer: &str, content_hash: &str, ver_version: &str) -> String {
    uuid_text(&uuid5(&record_namespace(domain), &format!("{producer}|{content_hash}|{ver_version}")))
}

/// A fresh random-looking UUIDv4 (for minting a producer identity): SHA-256 of the clock, the process, a counter and
/// `salt`. Unique, not secret, which is all a producer id needs.
pub(crate) fn uuid4(salt: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let d = Sha256::digest(format!("{nanos}|{}|{n}|{salt}", std::process::id()).as_bytes());
    let mut u = [0u8; 16];
    u.copy_from_slice(d.get(..16).unwrap_or(&[0; 16]));
    u[6] = (u[6] & 0x0f) | 0x40;
    u[8] = (u[8] & 0x3f) | 0x80;
    uuid_text(&u)
}

/// Now, as RFC 3339 UTC with seconds (`2026-10-11T09:30:00Z`).
pub(crate) fn now() -> String {
    format!("{}Z", crate::import::system_clock())
}

/// A catalog time (`YYYY-MM-DDTHH:MM:SS`, which LightCraft writes in UTC: [`crate::import::system_clock`]) as RFC
/// 3339, never later than `not_after` (a clock that ran ahead must not make the chain run backwards). `None` when it
/// does not parse.
pub(crate) fn catalog_time(t: &str, not_after: &str) -> Option<String> {
    let secs = lightcraft_catalog::dates::iso_seconds(t.get(..19)?)?;
    let cap = lightcraft_catalog::dates::iso_seconds(not_after.get(..19)?)?;
    Some(format!("{}Z", lightcraft_catalog::dates::civil(secs.min(cap))))
}

/// A `file://` URI for an absolute path (RFC 8089), percent-encoding what a URI cannot carry.
pub(crate) fn file_uri(path: &std::path::Path) -> String {
    let abs =
        if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir().map(|d| d.join(path)).unwrap_or_else(|_| path.to_path_buf()) };
    let text = abs.to_string_lossy().replace('\\', "/");
    let mut out = String::from("file://");
    if !text.starts_with('/') {
        out.push('/');
    }
    for b in text.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' | b':' => out.push(char::from(b)),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn base64_round_trips_and_matches_rfc4648() {
        assert_eq!(b64(b""), "");
        assert_eq!(b64(b"f"), "Zg==");
        assert_eq!(b64(b"fo"), "Zm8=");
        assert_eq!(b64(b"foobar"), "Zm9vYmFy");
        assert_eq!(b64url(&[0xfb, 0xff]), "-_8");
        for n in 0..40u8 {
            let v: Vec<u8> = (0..n).map(|i| i.wrapping_mul(37)).collect();
            assert_eq!(unb64(&b64(&v)).unwrap(), v);
        }
        assert!(unb64("a*b").is_none());
    }

    #[test]
    fn jcs_sorts_escapes_and_prints_numbers_as_ecmascript() {
        let v = json!({"b": [1, 0.5, -0.0, 1e21, 1e-7, 123.456, 0.00392156862745098], "a": "é\"\n\u{1}", "\u{20ac}": 1, "\u{1f600}": 2});
        // U+20AC sorts after U+1F600 by UTF-16 code units (0xD83D < 0x20AC is false: 0x20AC < 0xD83D)
        assert_eq!(
            jcs(&v).unwrap(),
            "{\"a\":\"é\\\"\\n\\u0001\",\"b\":[1,0.5,0,1e+21,1e-7,123.456,0.00392156862745098],\"\u{20ac}\":1,\"\u{1f600}\":2}"
        );
        assert_eq!(jcs(&json!(1e20)).unwrap(), "100000000000000000000");
        assert_eq!(jcs(&json!(0.000001)).unwrap(), "0.000001");
        assert_eq!(jcs(&json!(-2.5e-9)).unwrap(), "-2.5e-9");
    }

    #[test]
    fn uuid5_matches_the_reference_namespace() {
        // uuid5(NAMESPACE_DNS, "verstandard.org"), the reference producer's RECORD_ID_NAMESPACE
        assert_eq!(uuid_text(&record_namespace("verstandard.org")), "26d2cc27-b556-5ef2-a656-86c0ac2e63a0");
        // RFC 9562 test vector: uuid5(DNS, "www.example.com")
        assert_eq!(uuid_text(&uuid5(&NAMESPACE_DNS, "www.example.com")), "2ed6657d-e927-568b-95e1-2665a8aea6a2");
        let a = uuid4("x");
        assert_ne!(a, uuid4("x"));
        assert_eq!(a.as_bytes()[14], b'4');
    }

    #[test]
    fn times_and_uris() {
        assert_eq!(catalog_time("2026-09-28T09:30:00", "2026-10-11T00:00:00Z").as_deref(), Some("2026-09-28T09:30:00Z"));
        assert_eq!(catalog_time("2030-01-01T00:00:00", "2026-10-11T00:00:00Z").as_deref(), Some("2026-10-11T00:00:00Z"));
        assert_eq!(catalog_time("yesterday", "2026-10-11T00:00:00Z"), None);
        assert_eq!(file_uri(std::path::Path::new("/a b/ü.jpg")), "file:///a%20b/%C3%BC.jpg");
        assert!(now().ends_with('Z'));
        assert_eq!(unhex("0aff").unwrap(), vec![10, 255]);
        assert!(unhex("0g").is_none());
    }
}
