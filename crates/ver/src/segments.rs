//! Byte-exact metadata segments (errata E11) and per-family availability (E21).
//!
//! What each container's "segment" is, per E11's table: a JPEG `APPn` marker's payload
//! (identifier prefix included, marker and length excluded), a PNG chunk as stored (length,
//! type, data, CRC; compressed data stays compressed), a WebP RIFF chunk's payload (no pad
//! byte), a GIF Application Extension block as stored, a TIFF IFD entry's value bytes, a BMP
//! V5 embedded profile. One entry per stored segment, in file order (P2, P3). TIFF EXIF has no
//! defined range in 1.0.x: availability `unknown` when an Exif IFD exists, never a `raw[]` entry.

use crate::cpnp::Format;
use crate::tiffifd::Ifd;

/// An availability state (the schema's `availabilityState` enum; `UpstreamWithheld` is 1.1-draft only).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Present,
    Absent,
    StrippedSuspected,
    Redacted,
    Unknown,
    UpstreamWithheld,
}

impl State {
    /// The schema token.
    pub fn as_str(self) -> &'static str {
        match self {
            State::Present => "present",
            State::Absent => "absent",
            State::StrippedSuspected => "stripped_suspected",
            State::Redacted => "redacted",
            State::Unknown => "unknown",
            State::UpstreamWithheld => "upstream_withheld",
        }
    }

    /// The state for a schema token.
    pub fn from_token(s: &str) -> Option<State> {
        Some(match s {
            "present" => State::Present,
            "absent" => State::Absent,
            "stripped_suspected" => State::StrippedSuspected,
            "redacted" => State::Redacted,
            "unknown" => State::Unknown,
            "upstream_withheld" => State::UpstreamWithheld,
            _ => return None,
        })
    }
}

/// `metadata.availability`: one state per keyed family.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Availability {
    pub exif: State,
    pub xmp: State,
    pub iptc: State,
    pub icc: State,
    pub c2pa: State,
}

impl Default for Availability {
    fn default() -> Self {
        Availability { exif: State::Unknown, xmp: State::Unknown, iptc: State::Unknown, icc: State::Unknown, c2pa: State::Unknown }
    }
}

impl Availability {
    /// The `metadata.availability` object.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "exif": self.exif.as_str(),
            "xmp": self.xmp.as_str(),
            "iptc": self.iptc.as_str(),
            "icc": self.icc.as_str(),
            "c2pa": self.c2pa.as_str(),
        })
    }
}

/// One stored segment, as preserved: `family` is `"exif" | "xmp" | "iptc-iim" | "icc" | "c2pa" | "other"`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawSegment {
    pub family: &'static str,
    pub bytes: Vec<u8>,
}

/// JPEG markers before the first SOS, as `(marker, payload)` (payload excludes marker and length).
pub(crate) fn jpeg_markers(b: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    if b.get(..2) != Some(&[0xFF, 0xD8][..]) {
        return out;
    }
    let mut p = 2usize;
    loop {
        while b.get(p).is_some_and(|&c| c != 0xFF) {
            p += 1;
        }
        while b.get(p) == Some(&0xFF) {
            p += 1;
        }
        let Some(&m) = b.get(p) else { break };
        p += 1;
        if matches!(m, 0xD8 | 0x01 | 0xD0..=0xD7) {
            continue;
        }
        if m == 0xD9 || m == 0xDA {
            break;
        }
        let Some(len) = b.get(p..p + 2).map(|l| usize::from(l[0]) << 8 | usize::from(l[1])) else { break };
        if len < 2 {
            break;
        }
        let Some(payload) = b.get(p + 2..p + len) else { break };
        out.push((m, payload));
        p += len;
    }
    out
}

/// PNG chunks as `(type, data)`, `None` when the signature or chunk framing is broken.
pub(crate) fn png_chunks(b: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    Some(png_chunks_framed(b)?.into_iter().map(|(t, d, _)| (t, d)).collect())
}

/// PNG chunks as `(type, data, whole chunk as stored)`.
fn png_chunks_framed(b: &[u8]) -> Option<Vec<([u8; 4], &[u8], &[u8])>> {
    if b.get(..8)? != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let mut out = Vec::new();
    let mut p = 8usize;
    while p < b.len() {
        let len = usize::try_from(u32::from_be_bytes(b.get(p..p + 4)?.try_into().ok()?)).ok()?;
        let mut typ = [0u8; 4];
        typ.copy_from_slice(b.get(p + 4..p + 8)?);
        let end = p.checked_add(12)?.checked_add(len)?;
        let data = b.get(p + 8..p + 8 + len)?;
        let whole = b.get(p..end)?;
        out.push((typ, data, whole));
        p = end;
        if &typ == b"IEND" {
            break;
        }
    }
    Some(out)
}

/// The stored metadata segments of `bytes` and the availability they establish.
pub fn segments(bytes: &[u8], format: Format) -> (Availability, Vec<RawSegment>) {
    let segs = match format {
        Format::Jpeg => jpeg(bytes),
        Format::Png => png(bytes),
        Format::Webp => webp(bytes),
        Format::Gif => gif(bytes),
        Format::Bmp => bmp(bytes),
        Format::Tiff => return tiff(bytes),
        Format::Avif | Format::Heic => return (Availability::default(), Vec::new()),
    };
    let Some(segs) = segs else {
        // Unparseable container structure: the families cannot be told apart.
        return (Availability::default(), Vec::new());
    };
    let state = |f: &str| if segs.iter().any(|s| s.family == f) { State::Present } else { State::Absent };
    let mut avail = Availability { exif: state("exif"), xmp: state("xmp"), iptc: state("iptc-iim"), icc: state("icc"), c2pa: state("c2pa") };
    if format == Format::Bmp {
        // BMP carries no family but ICC in 1.0.x scope; C2PA has no defined BMP carriage.
        avail.c2pa = State::Unknown;
    }
    (avail, segs)
}

fn seg(family: &'static str, bytes: &[u8]) -> RawSegment {
    RawSegment { family, bytes: bytes.to_vec() }
}

fn jpeg(b: &[u8]) -> Option<Vec<RawSegment>> {
    if b.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    let mut out = Vec::new();
    for (m, payload) in jpeg_markers(b) {
        let family = match m {
            0xE1 if payload.starts_with(b"Exif\0\0") => "exif",
            0xE1 if payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0") => "xmp",
            0xE1 if payload.starts_with(b"http://ns.adobe.com/xmp/extension/\0") => "xmp",
            0xE2 if payload.starts_with(b"ICC_PROFILE\0") => "icc",
            0xED if payload.starts_with(b"Photoshop 3.0\0") && contains(payload, b"8BIM\x04\x04") => "iptc-iim",
            0xEB if payload.starts_with(b"JP") && contains(payload.get(..payload.len().min(96)).unwrap_or(&[]), b"c2pa") => "c2pa",
            0xE0..=0xEF | 0xFE => "other",
            _ => continue,
        };
        out.push(seg(family, payload));
    }
    Some(out)
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

fn png(b: &[u8]) -> Option<Vec<RawSegment>> {
    let mut out = Vec::new();
    for (typ, data, whole) in png_chunks_framed(b)? {
        let keyword = data.iter().position(|&c| c == 0).and_then(|n| data.get(..n)).unwrap_or(&[]);
        let family = match &typ {
            b"eXIf" => "exif",
            b"iCCP" => "icc",
            b"caBX" => "c2pa",
            b"iTXt" if keyword == b"XML:com.adobe.xmp" => "xmp",
            b"tEXt" | b"zTXt" | b"iTXt" => match keyword {
                b"Raw profile type exif" | b"Raw profile type APP1" => "exif",
                b"Raw profile type iptc" => "iptc-iim",
                b"Raw profile type xmp" => "xmp",
                b"Raw profile type icc" | b"Raw profile type icm" => "icc",
                _ => "other",
            },
            _ => continue,
        };
        out.push(seg(family, whole));
    }
    Some(out)
}

fn webp(b: &[u8]) -> Option<Vec<RawSegment>> {
    if b.get(..4)? != b"RIFF" || b.get(8..12)? != b"WEBP" {
        return None;
    }
    let mut out = Vec::new();
    let mut p = 12usize;
    while p + 8 <= b.len() {
        let fourcc = b.get(p..p + 4)?;
        let size = usize::try_from(u32::from_le_bytes(b.get(p + 4..p + 8)?.try_into().ok()?)).ok()?;
        let payload = b.get(p + 8..p.checked_add(8)?.checked_add(size)?)?;
        let family = match fourcc {
            b"EXIF" => Some("exif"),
            b"XMP " => Some("xmp"),
            b"ICCP" => Some("icc"),
            b"C2PA" => Some("c2pa"),
            _ => None,
        };
        if let Some(f) = family {
            out.push(seg(f, payload));
        }
        p = p + 8 + size + (size & 1);
    }
    Some(out)
}

fn gif(b: &[u8]) -> Option<Vec<RawSegment>> {
    if !(b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a")) {
        return None;
    }
    let flags = *b.get(10)?;
    let mut p = 13usize;
    if flags & 0x80 != 0 {
        p += 3 << ((flags & 7) + 1);
    }
    let end_of_subblocks = |mut q: usize| -> Option<usize> {
        loop {
            let n = usize::from(*b.get(q)?);
            q += 1;
            if n == 0 {
                return Some(q);
            }
            q += n;
        }
    };
    let mut out = Vec::new();
    loop {
        match b.get(p)? {
            0x21 => {
                let label = *b.get(p + 1)?;
                let end = end_of_subblocks(p + 2)?;
                if label == 0xFF {
                    let block = b.get(p..end)?;
                    let ident = b.get(p + 3..p + 14).unwrap_or(&[]);
                    let family = match ident {
                        b"XMP DataXMP" => "xmp",
                        b"ICCRGBG1012" => "icc",
                        b"NETSCAPE2.0" | b"ANIMEXTS1.0" => {
                            p = end;
                            continue;
                        }
                        _ => "other",
                    };
                    out.push(seg(family, block));
                }
                p = end;
            }
            0x2C => {
                let f = *b.get(p + 9)?;
                p += 10;
                if f & 0x80 != 0 {
                    p += 3 << ((f & 7) + 1);
                }
                p = end_of_subblocks(p + 1)?;
            }
            0x3B => break,
            _ => return None,
        }
    }
    Some(out)
}

fn bmp(b: &[u8]) -> Option<Vec<RawSegment>> {
    if b.get(..2)? != b"BM" {
        return None;
    }
    let le32 = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?)) };
    let header = 14usize;
    let size = le32(header)?;
    let mut out = Vec::new();
    if size >= 124 && b.get(header + 56..header + 60)? == b"DEBM" {
        // bV5CSType 'MBED' (stored little-endian as "DEBM"); data offset is from the header start.
        let off = usize::try_from(le32(header + 112)?).ok()?;
        let len = usize::try_from(le32(header + 116)?).ok()?;
        let start = header.checked_add(off)?;
        out.push(seg("icc", b.get(start..start.checked_add(len)?)?));
    }
    Some(out)
}

fn tiff(b: &[u8]) -> (Availability, Vec<RawSegment>) {
    let Some(ifd) = Ifd::parse(b) else { return (Availability::default(), Vec::new()) };
    let mut out = Vec::new();
    let mut avail = Availability {
        // E11: TIFF EXIF has no defined byte range; unknown when an Exif IFD exists.
        exif: if ifd.get(34665).is_some() { State::Unknown } else { State::Absent },
        xmp: State::Absent,
        iptc: State::Absent,
        icc: State::Absent,
        c2pa: State::Absent,
    };
    for e in &ifd.entries {
        let family = match e.tag {
            34675 => "icc",
            700 => "xmp",
            33723 => "iptc-iim",
            // C2PA's TIFF/DNG carriage tag.
            52545 => "c2pa",
            _ => continue,
        };
        match family {
            "icc" => avail.icc = State::Present,
            "xmp" => avail.xmp = State::Present,
            "iptc-iim" => avail.iptc = State::Present,
            _ => avail.c2pa = State::Present,
        }
        out.push(seg(family, e.value));
    }
    (avail, out)
}

/// Schema-valid `metadata.raw[]` entries for `segs`, in order: `{segment, bytes, sha256}` with
/// standard padded base64 and the SHA-256 of the decoded octets (E11).
pub fn raw_entries(segs: &[RawSegment]) -> Vec<serde_json::Value> {
    segs.iter()
        .map(|s| {
            serde_json::json!({
                "segment": s.family,
                "bytes": crate::b64::encode(&s.bytes),
                "sha256": crate::hashing::sha256_hex(&s.bytes),
            })
        })
        .collect()
}
