//! Signing a Record (errata E17): a detached JWS (RFC 7515 Appendix F) with EdDSA (Ed25519, RFC 8037) over the
//! RFC 8785 JCS form of the Record with `/provenance/signature` removed, exactly as the upstream validator's
//! `signature_stage.py` re-derives and verifies it.
//!
//! The key is an unencrypted PKCS#8 Ed25519 private key in PEM (`openssl genpkey -algorithm ed25519`), named by
//! `ver.config {signKeyPath}`; it is read when a Record is signed and never copied anywhere.

use ed25519_dalek::{Signer as _, SigningKey};
use serde_json::{Value, json};

use super::util::{b64url, jcs, sha256_hex, unb64};

/// The DER prefix of a PKCS#8 `PrivateKeyInfo` holding an Ed25519 seed (RFC 8410 §7): version 0, algorithm
/// 1.3.101.112, then the 32-byte seed as an OCTET STRING inside an OCTET STRING.
const PKCS8_ED25519_PREFIX: [u8; 16] = [0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20];
/// The DER prefix of a `SubjectPublicKeyInfo` holding an Ed25519 public key (RFC 8410 §4).
#[cfg(test)]
const SPKI_ED25519_PREFIX: [u8; 12] = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
/// Largest key file read.
const MAX_KEY_FILE: u64 = 64 * 1024;

/// A loaded signing key.
#[derive(Clone)]
pub(crate) struct Signer {
    key: SigningKey,
    /// `ed25519:` and the first 16 hex of SHA-256 over the raw public key: names the key without revealing anything.
    pub key_id: String,
}

impl std::fmt::Debug for Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Signer").field("key_id", &self.key_id).finish_non_exhaustive()
    }
}

impl Signer {
    /// The key in the PEM file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Signer, String> {
        use std::io::Read;
        let mut text = String::new();
        std::fs::File::open(path)
            .and_then(|f| f.take(MAX_KEY_FILE).read_to_string(&mut text))
            .map_err(|e| format!("signing key {}: {e}", path.display()))?;
        Signer::from_pem(&text).map_err(|e| format!("signing key {}: {e}", path.display()))
    }

    /// The key in a PKCS#8 PEM text.
    pub fn from_pem(text: &str) -> Result<Signer, String> {
        let body =
            pem_body(text, "PRIVATE KEY").ok_or("not a PEM \"PRIVATE KEY\" (PKCS#8); an Ed25519 key from `openssl genpkey -algorithm ed25519` is")?;
        let der = unb64(&body).ok_or("the PEM body is not base64")?;
        let seed = der.strip_prefix(&PKCS8_ED25519_PREFIX[..]).filter(|s| s.len() == 32).ok_or("not an unencrypted Ed25519 PKCS#8 key")?;
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(seed);
        Ok(Signer::from_seed(&bytes))
    }

    pub fn from_seed(seed: &[u8; 32]) -> Signer {
        let key = SigningKey::from_bytes(seed);
        let public = key.verifying_key().to_bytes();
        let key_id = format!("ed25519:{}", sha256_hex(&public).get(..16).unwrap_or(""));
        Signer { key, key_id }
    }

    /// The public key as a PEM `SubjectPublicKeyInfo` (what a verifier is given).
    #[cfg(test)]
    pub fn public_pem(&self) -> String {
        let mut der = SPKI_ED25519_PREFIX.to_vec();
        der.extend_from_slice(&self.key.verifying_key().to_bytes());
        format!("-----BEGIN PUBLIC KEY-----\n{}\n-----END PUBLIC KEY-----\n", super::util::b64(&der))
    }

    /// Sign `record` in place: `provenance.signature` is replaced by a detached JWS over the record without it.
    pub fn sign(&self, record: &mut Value) -> Result<(), String> {
        if let Some(p) = record.get_mut("provenance").and_then(Value::as_object_mut) {
            p.remove("signature");
        }
        let payload = jcs(record)?;
        let header = jcs(&json!({"alg": "EdDSA", "kid": self.key_id}))?;
        let header_b64 = b64url(header.as_bytes());
        let input = format!("{header_b64}.{}", b64url(payload.as_bytes()));
        let sig = self.key.sign(input.as_bytes());
        let value = format!("{header_b64}..{}", b64url(&sig.to_bytes()));
        let provenance = record.as_object_mut().ok_or("a record is an object")?.entry("provenance").or_insert_with(|| json!({}));
        let p = provenance.as_object_mut().ok_or("provenance is not an object")?;
        p.insert("signature".into(), json!({"format": "jws-detached", "canonicalization": "rfc8785-jcs", "value": value, "key_id": self.key_id}));
        Ok(())
    }
}

/// The base64 body of the first PEM block labelled `label`.
fn pem_body(text: &str, label: &str) -> Option<String> {
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let start = text.find(&begin)? + begin.len();
    let stop = start + text.get(start..)?.find(&end)?;
    Some(text.get(start..stop)?.chars().filter(|c| !c.is_whitespace()).collect())
}

/// A verifier's check of a signature made by [`Signer::sign`] (the tests'): `Ok(true)` when it verifies under
/// `public` (32 raw bytes).
#[cfg(test)]
pub(crate) fn verify(record: &Value, public: &[u8; 32]) -> Result<bool, String> {
    use ed25519_dalek::{Signature, Verifier as _, VerifyingKey};
    let value = record.pointer("/provenance/signature/value").and_then(Value::as_str).ok_or("no signature")?;
    let mut parts = value.split('.');
    let (Some(header), Some(""), Some(sig), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err("not a detached JWS".into());
    };
    let mut stripped = record.clone();
    if let Some(p) = stripped.get_mut("provenance").and_then(Value::as_object_mut) {
        p.remove("signature");
    }
    let input = format!("{header}.{}", b64url(jcs(&stripped)?.as_bytes()));
    let sig = unb64url(sig).ok_or("signature is not base64url")?;
    let sig = Signature::from_slice(&sig).map_err(|e| e.to_string())?;
    let key = VerifyingKey::from_bytes(public).map_err(|e| e.to_string())?;
    Ok(key.verify(input.as_bytes(), &sig).is_ok())
}

#[cfg(test)]
fn unb64url(s: &str) -> Option<Vec<u8>> {
    unb64(&s.replace('-', "+").replace('_', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_round_trip_and_pem() {
        let s = Signer::from_seed(&[7; 32]);
        let mut der = PKCS8_ED25519_PREFIX.to_vec();
        der.extend_from_slice(&[7; 32]);
        let pem = format!("-----BEGIN PRIVATE KEY-----\n{}\n-----END PRIVATE KEY-----\n", super::super::util::b64(&der));
        let loaded = Signer::from_pem(&pem).unwrap();
        assert_eq!(loaded.key_id, s.key_id);
        assert!(Signer::from_pem("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----").is_err());
        assert!(Signer::from_pem("hello").is_err());
        let mut r = json!({"b": 1, "a": [0.5, "x"], "provenance": {"chain": []}});
        s.sign(&mut r).unwrap();
        let public = s.key.verifying_key().to_bytes();
        assert!(verify(&r, &public).unwrap());
        assert!(r["provenance"]["signature"]["value"].as_str().unwrap().contains(".."));
        r["b"] = json!(2);
        assert!(!verify(&r, &public).unwrap());
        assert!(s.public_pem().starts_with("-----BEGIN PUBLIC KEY-----"));
    }
}
