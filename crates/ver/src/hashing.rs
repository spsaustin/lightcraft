//! The identity gradient: content, pixel, perceptual.

use crate::cpnp::{self, CanonicalBuffer, CpnpError};
pub use lightcraft_vera::digest::Planes;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Lowercase hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(HEX[usize::from(b >> 4)]));
        s.push(char::from(HEX[usize::from(b & 15)]));
    }
    s
}

/// SHA-256 of `bytes` as 64 lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// `content_hash`: SHA-256 over the Asset's bytes, 64 lowercase hex.
pub fn content_hash(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// `pixel_hash`: SHA-256 over `b"VER1:" + w + b":" + h + b":sRGB8:" + rgb` (errata E10).
pub fn pixel_hash(buf: &CanonicalBuffer) -> String {
    let mut h = Sha256::new();
    h.update(b"VER1:");
    h.update(buf.width.to_string().as_bytes());
    h.update(b":");
    h.update(buf.height.to_string().as_bytes());
    h.update(b":sRGB8:");
    h.update(&buf.rgb);
    hex(&h.finalize())
}

/// The Canonical Buffer's luma planes, built in one pass and shared by PDQ, its quality figure and
/// pHash (errata E22: over the Canonical Buffer, at its own resolution, never a thumbnail).
/// `None` only for an empty buffer.
pub fn luma(buf: &CanonicalBuffer) -> Option<Planes> {
    Planes::from_rgb8(buf.width as usize, buf.height as usize, &buf.rgb)
}

/// The perceptual digests of one Canonical Buffer, from one set of [`luma`] planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Perceptual {
    /// `pdq/1.0`, 64 lowercase hex in Meta's reference format.
    pub pdq: String,
    /// PDQ's 0..=100 image-domain quality figure.
    pub pdq_quality: u8,
    /// `phash-dct/64`, 16 lowercase hex in `imagehash`'s format.
    pub phash: String,
}

/// Both perceptual digests and PDQ's quality from prepared planes (zeros for an empty buffer).
pub fn perceptual_from(planes: Option<&Planes>) -> Perceptual {
    let (bits, pdq_quality) = planes.and_then(lightcraft_vera::digest::pdq_planes).unwrap_or(([0; 32], 0));
    let phash = planes.and_then(lightcraft_vera::digest::phash_planes).unwrap_or(0);
    Perceptual { pdq: hex(&bits), pdq_quality, phash: format!("{phash:016x}") }
}

/// Both perceptual digests and PDQ's quality of the Canonical Buffer, the luma built once.
pub fn perceptual(buf: &CanonicalBuffer) -> Perceptual {
    perceptual_from(luma(buf).as_ref())
}

/// `pdq/1.0` over the Canonical Buffer, 64 lowercase hex in Meta's reference format
/// (bit-exact with the reference on 8-bit RGB). Prefer [`perceptual`] when pHash is wanted too.
pub fn pdq(buf: &CanonicalBuffer) -> String {
    pdq_from(luma(buf).as_ref())
}

/// [`pdq`] from prepared planes.
pub fn pdq_from(planes: Option<&Planes>) -> String {
    hex(&planes.and_then(lightcraft_vera::digest::pdq_planes).map_or([0; 32], |(b, _)| b))
}

/// PDQ's 0..=100 quality figure of the Canonical Buffer (the reference's image-domain metric).
pub fn pdq_quality(buf: &CanonicalBuffer) -> u8 {
    luma(buf).as_ref().and_then(lightcraft_vera::digest::pdq_planes).map_or(0, |(_, q)| q)
}

/// `phash-dct/64` over the Canonical Buffer, 16 lowercase hex in `imagehash`'s format (bit-exact
/// with `imagehash.phash`). Prefer [`perceptual`] when PDQ is wanted too.
pub fn phash_dct64(buf: &CanonicalBuffer) -> String {
    phash_from(luma(buf).as_ref())
}

/// [`phash_dct64`] from prepared planes.
pub fn phash_from(planes: Option<&Planes>) -> String {
    format!("{:016x}", planes.and_then(lightcraft_vera::digest::phash_planes).unwrap_or(0))
}

/// Every identity value of one Asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub content_hash: String,
    pub pixel_hash: String,
    pub width: u32,
    pub height: u32,
    pub pdq: String,
    pub phash: String,
}

/// Canonicalise `bytes` and compute every identity value.
pub fn identity(bytes: &[u8]) -> Result<(CanonicalBuffer, Identity), CpnpError> {
    identity_with_quality(bytes).map(|(buf, id, _)| (buf, id))
}

/// [`identity`] plus PDQ's quality figure, every digest from one decode and one luma pass.
pub fn identity_with_quality(bytes: &[u8]) -> Result<(CanonicalBuffer, Identity, u8), CpnpError> {
    let buf = cpnp::canonicalize(bytes)?;
    let p = perceptual(&buf);
    let id = Identity {
        content_hash: content_hash(bytes),
        pixel_hash: pixel_hash(&buf),
        width: buf.width,
        height: buf.height,
        pdq: p.pdq,
        phash: p.phash,
    };
    Ok((buf, id, p.pdq_quality))
}
