//! A minimal, bounded reader for the first IFD of a TIFF structure (a TIFF file or an EXIF blob).
//!
//! CPNP-1 reads exactly one thing from EXIF (tag 274 in IFD0, errata E6) and E11 preserves TIFF
//! tag value bytes as stored, so this reads IFD0's entries and hands back each entry's value
//! bytes as they sit in the file. Nothing here follows IFD chains or sub-IFDs except to report
//! that an Exif IFD pointer (34665) exists.

/// One IFD entry: tag, field type, count and the stored value bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry<'a> {
    pub tag: u16,
    pub typ: u16,
    pub count: u64,
    pub value: &'a [u8],
}

/// IFD0 of a TIFF structure.
#[derive(Clone, Debug)]
pub(crate) struct Ifd<'a> {
    pub big_endian: bool,
    pub entries: Vec<Entry<'a>>,
}

/// Octets per value of a TIFF field type (0 for unknown types).
pub(crate) fn type_size(typ: u16) -> u64 {
    match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => 0,
    }
}

impl<'a> Ifd<'a> {
    /// Parse IFD0 of `data` (`II*\0`, `MM\0*` or BigTIFF). `None` when it is not TIFF-shaped.
    pub(crate) fn parse(data: &'a [u8]) -> Option<Ifd<'a>> {
        let be = match data.get(..2)? {
            b"II" => false,
            b"MM" => true,
            _ => return None,
        };
        let u16_at = |at: usize| -> Option<u16> {
            let b = data.get(at..at.checked_add(2)?)?;
            Some(if be { u16::from_be_bytes([b[0], b[1]]) } else { u16::from_le_bytes([b[0], b[1]]) })
        };
        let u32_at = |at: usize| -> Option<u32> {
            let b = data.get(at..at.checked_add(4)?)?;
            let a = [b[0], b[1], b[2], b[3]];
            Some(if be { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) })
        };
        let u64_at = |at: usize| -> Option<u64> {
            let b = data.get(at..at.checked_add(8)?)?;
            let mut a = [0u8; 8];
            a.copy_from_slice(b);
            Some(if be { u64::from_be_bytes(a) } else { u64::from_le_bytes(a) })
        };
        let magic = u16_at(2)?;
        let big = match magic {
            42 => false,
            43 => true,
            _ => return None,
        };
        let (ifd, count, entry_size, header) = if big {
            let off = usize::try_from(u64_at(8)?).ok()?;
            (off, usize::try_from(u64_at(off)?).ok()?, 20usize, 8usize)
        } else {
            let off = usize::try_from(u32_at(4)?).ok()?;
            (off, usize::from(u16_at(off)?), 12usize, 2usize)
        };
        let inline = if big { 8u64 } else { 4u64 };
        let mut entries = Vec::new();
        for i in 0..count.min(4096) {
            let at = ifd.checked_add(header)?.checked_add(i.checked_mul(entry_size)?)?;
            let Some(tag) = u16_at(at) else { break };
            let Some(typ) = u16_at(at + 2) else { break };
            let cnt = if big { u64_at(at + 4) } else { u32_at(at + 4).map(u64::from) };
            let Some(cnt) = cnt else { break };
            let len = type_size(typ).checked_mul(cnt);
            let field = at + if big { 12 } else { 8 };
            let value = match len {
                Some(len) if len <= inline => usize::try_from(len).ok().and_then(|l| data.get(field..field + l)),
                Some(len) => {
                    let off = if big { u64_at(field) } else { u32_at(field).map(u64::from) };
                    off.and_then(|o| {
                        let s = usize::try_from(o).ok()?;
                        let l = usize::try_from(len).ok()?;
                        data.get(s..s.checked_add(l)?)
                    })
                }
                None => None,
            };
            if let Some(value) = value {
                entries.push(Entry { tag, typ, count: cnt, value });
            }
        }
        Some(Ifd { big_endian: be, entries })
    }

    pub(crate) fn get(&self, tag: u16) -> Option<&Entry<'a>> {
        self.entries.iter().find(|e| e.tag == tag)
    }

    /// Unsigned integer values of an entry (BYTE, SHORT, LONG, LONG8).
    pub(crate) fn uints(&self, tag: u16) -> Option<Vec<u64>> {
        let e = self.get(tag)?;
        let size = type_size(e.typ) as usize;
        if !matches!(e.typ, 1 | 3 | 4 | 16) || size == 0 {
            return None;
        }
        Some(
            e.value
                .chunks_exact(size)
                .map(|c| match (size, self.big_endian) {
                    (1, _) => u64::from(c[0]),
                    (2, true) => u64::from(u16::from_be_bytes([c[0], c[1]])),
                    (2, false) => u64::from(u16::from_le_bytes([c[0], c[1]])),
                    (4, true) => u64::from(u32::from_be_bytes([c[0], c[1], c[2], c[3]])),
                    (4, false) => u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    (_, true) => u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
                    (_, false) => u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
                })
                .collect(),
        )
    }
}

/// The CPNP-1 orientation of an EXIF blob (with or without the `Exif\0\0` prefix): tag 274 in
/// IFD0, type SHORT, count 1, value 1–8; anything else is 1 (errata E6).
pub(crate) fn exif_orientation(exif: &[u8]) -> u16 {
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    Ifd::parse(tiff).map(|ifd| orientation(&ifd)).unwrap_or(1)
}

/// Tag 274 of a parsed IFD0, by the E6 rules.
pub(crate) fn orientation(ifd: &Ifd) -> u16 {
    let Some(e) = ifd.get(274) else { return 1 };
    if e.typ != 3 || e.count != 1 {
        return 1;
    }
    match ifd.uints(274).and_then(|v| v.first().copied()) {
        Some(v @ 1..=8) => v as u16,
        _ => 1,
    }
}
