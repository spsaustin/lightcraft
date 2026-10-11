//! A JPEG decoder whose output is libjpeg-turbo's, bit for bit, for the paths CPNP-1 pins.
//!
//! The profile `verd-pillow/1` decodes JPEG through Pillow's bundled libjpeg-turbo with its
//! defaults: the integer "slow" IDCT (`JDCT_ISLOW`, `jidctint.c`), "fancy" triangle-filter chroma
//! upsampling (`jdsample.c`: `h2v1`, `h2v2`, `h1v2`; other ratios replicate), and the fixed-point
//! YCbCr → RGB tables of `jdcolor.c`. Neither pure-Rust decoder in the workspace reproduces all
//! three (`zune-jpeg` and `jpeg-decoder` both carry stb_image's IDCT, and `jpeg-decoder`'s h2v2
//! rounds both output phases with `+8`), so this module implements them from the published
//! algorithms (the IJG/libjpeg-turbo documentation of the integer IDCT's constants and the
//! triangle filter's weights), not by translating either library.
//!
//! Scope: baseline and extended sequential Huffman (SOF0/SOF1) and progressive Huffman (SOF2),
//! 8-bit samples, 1, 3 or 4 components, any sampling factors 1–4, restart intervals. Arithmetic
//! coding, lossless, hierarchical and 12-bit JPEG are refused ([`Error::Unsupported`]).
//!
//! Damage follows libjpeg: entropy data that stops at a marker decodes the rest of the segment
//! as zero coefficients (uniform grey); data that stops at end of file is an error (Pillow raises
//! "image file is truncated" there).

/// What went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Error {
    Unsupported(String),
    Malformed(String),
}

/// Decoded samples in the colour space libjpeg hands Pillow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Samples {
    /// One channel ("L").
    Gray(Vec<u8>),
    /// Interleaved RGB.
    Rgb(Vec<u8>),
    /// Interleaved CMYK as libjpeg outputs it (Adobe-inverted for Adobe files; Pillow inverts).
    Cmyk(Vec<u8>),
}

#[derive(Debug, Clone)]
pub(crate) struct Image {
    pub width: u32,
    pub height: u32,
    pub samples: Samples,
}

/// Refuse anything larger than this many pixels (memory is ~6 bytes per pixel and component).
const MAX_PIXELS: u64 = 1 << 28;

/// Zigzag position → natural (row-major) index, with libjpeg's 16 safety entries past the end.
const NATURAL: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
    36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63,
];

fn natural(k: usize) -> usize {
    NATURAL.get(k).copied().unwrap_or(63)
}

#[derive(Clone, Debug, Default)]
struct Huffman {
    /// `maxcode[l]` for code length `l` (1..=16), -1 when no code of that length; `[17]` is the sentinel.
    maxcode: [i32; 18],
    valoffset: [i32; 18],
    values: Vec<u8>,
}

impl Huffman {
    fn build(counts: &[u8; 16], values: Vec<u8>, is_dc: bool) -> Result<Huffman, Error> {
        let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
        if total > 256 || total > values.len() {
            return Err(Error::Malformed("bad Huffman table".into()));
        }
        let mut h = Huffman { maxcode: [-1; 18], valoffset: [0; 18], values };
        let mut code: i32 = 0;
        let mut p: i32 = 0;
        for l in 1..=16usize {
            let n = i32::from(counts[l - 1]);
            if n > 0 {
                h.valoffset[l] = p - code;
                p += n;
                code += n;
                h.maxcode[l] = code - 1;
            }
            // A table whose codes overflow their length is invalid (libjpeg: JERR_BAD_HUFF_TABLE).
            if code > (1 << l) {
                return Err(Error::Malformed("bad Huffman table".into()));
            }
            code <<= 1;
        }
        h.maxcode[17] = 0xFFFFF;
        if is_dc && h.values.iter().take(total).any(|&v| v > 15) {
            return Err(Error::Malformed("bad DC Huffman table".into()));
        }
        Ok(h)
    }
}

#[derive(Clone, Debug)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Blocks per row / column covered by MCUs (interleaved scans).
    bw: usize,
    bh: usize,
    /// Blocks holding real samples (non-interleaved scans cover exactly these).
    real_bw: usize,
    real_bh: usize,
    /// Downsampled dimensions in samples (`ceil(image · samp / max_samp)`).
    dw: usize,
    dh: usize,
    coefs: Vec<[i16; 64]>,
    dc_table: usize,
    ac_table: usize,
}

struct Frame {
    progressive: bool,
    width: usize,
    height: usize,
    hmax: usize,
    vmax: usize,
    mcux: usize,
    mcuy: usize,
    comps: Vec<Component>,
}

/// Entropy-coded data reader with libjpeg's end-of-data behaviour.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
    /// A marker was reached: further bits read as zero.
    hit_marker: bool,
    /// libjpeg's `insufficient_data`: the remaining MCUs of this restart segment stay zero.
    insufficient: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Bits<'a> {
        Bits { data, pos, acc: 0, nbits: 0, hit_marker: false, insufficient: false }
    }

    /// Make at least `n` (≤ 25) bits available.
    fn fill(&mut self, n: u32) -> Result<(), Error> {
        while self.nbits < n {
            if self.hit_marker {
                // libjpeg pads with zeros once it has reached a marker and flags the segment.
                self.insufficient = true;
                self.acc <<= 8;
                self.nbits += 8;
                continue;
            }
            let Some(&b) = self.data.get(self.pos) else {
                return Err(Error::Malformed("image file is truncated".into()));
            };
            if b == 0xFF {
                // Skip fill bytes; FF00 is a stuffed FF, anything else is a marker.
                let mut q = self.pos + 1;
                while self.data.get(q) == Some(&0xFF) {
                    q += 1;
                }
                match self.data.get(q) {
                    None => return Err(Error::Malformed("image file is truncated".into())),
                    Some(0) => {
                        self.pos = q + 1;
                        self.acc = (self.acc << 8) | 0xFF;
                        self.nbits += 8;
                    }
                    Some(_) => {
                        // Leave the position on the marker's first FF.
                        self.pos = q - 1;
                        self.hit_marker = true;
                    }
                }
            } else {
                self.pos += 1;
                self.acc = (self.acc << 8) | u64::from(b);
                self.nbits += 8;
            }
        }
        Ok(())
    }

    fn bits(&mut self, n: u32) -> Result<i32, Error> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n)?;
        self.nbits -= n;
        Ok(((self.acc >> self.nbits) & ((1u64 << n) - 1)) as i32)
    }

    fn bit(&mut self) -> Result<i32, Error> {
        self.bits(1)
    }

    fn decode(&mut self, h: &Huffman) -> Result<u8, Error> {
        let mut code = 0i32;
        let mut l = 1usize;
        loop {
            code = (code << 1) | self.bit()?;
            if code <= h.maxcode[l] || l >= 17 {
                break;
            }
            l += 1;
        }
        if l >= 17 {
            // libjpeg: "Corrupt JPEG data: bad Huffman code" and a zero symbol.
            return Ok(0);
        }
        let idx = code + h.valoffset[l];
        Ok(usize::try_from(idx).ok().and_then(|i| h.values.get(i)).copied().unwrap_or(0))
    }

    /// Discard buffered bits and step over the next marker if it is RSTn (restart processing).
    fn restart(&mut self) {
        self.acc = 0;
        self.nbits = 0;
        // Find the next marker.
        let mut p = self.pos;
        while p + 1 < self.data.len() {
            if self.data.get(p) == Some(&0xFF) {
                let mut q = p + 1;
                while self.data.get(q) == Some(&0xFF) {
                    q += 1;
                }
                match self.data.get(q) {
                    Some(0) => p = q + 1,
                    Some(m) if (0xD0..=0xD7).contains(m) => {
                        self.pos = q + 1;
                        self.hit_marker = false;
                        self.insufficient = false;
                        return;
                    }
                    Some(_) => {
                        self.pos = q - 1;
                        self.hit_marker = true;
                        return;
                    }
                    None => break,
                }
            } else {
                p += 1;
            }
        }
        self.pos = self.data.len();
        self.hit_marker = true;
    }
}

fn extend(r: i32, s: u32) -> i32 {
    if s == 0 {
        return 0;
    }
    if r < (1 << (s - 1)) { r + ((-1i32) << s) + 1 } else { r }
}

fn be16(b: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(*b.get(at)?) << 8 | usize::from(*b.get(at + 1)?))
}

/// Decode `bytes` as libjpeg-turbo with Pillow's settings would.
pub(crate) fn decode(bytes: &[u8]) -> Result<Image, Error> {
    if bytes.get(..2) != Some(&[0xFF, 0xD8][..]) {
        return Err(Error::Malformed("not a JPEG file".into()));
    }
    let mut pos = 2usize;
    let mut qt: [Option<[u16; 64]>; 4] = [None; 4];
    let mut dc: [Option<Huffman>; 4] = [None, None, None, None];
    let mut ac: [Option<Huffman>; 4] = [None, None, None, None];
    let mut restart_interval = 0usize;
    let mut frame: Option<Frame> = None;
    let mut saw_jfif = false;
    let mut adobe: Option<u8> = None;
    loop {
        // next_marker: skip garbage, then fill bytes.
        while bytes.get(pos).is_some_and(|&b| b != 0xFF) {
            pos += 1;
        }
        while bytes.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let Some(&marker) = bytes.get(pos) else {
            // No EOI: libjpeg warns and finishes when the image data is complete.
            break;
        };
        pos += 1;
        match marker {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xD9 => break,
            _ => {}
        }
        let len = be16(bytes, pos).ok_or_else(|| Error::Malformed("truncated marker segment".into()))?;
        if len < 2 {
            return Err(Error::Malformed("bad marker length".into()));
        }
        let seg = bytes.get(pos + 2..pos + len).ok_or_else(|| Error::Malformed("image file is truncated".into()))?;
        let seg_end = pos + len;
        match marker {
            0xC0..=0xC2 => {
                if frame.is_some() {
                    return Err(Error::Malformed("more than one frame".into()));
                }
                frame = Some(parse_sof(seg, marker == 0xC2)?);
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(Error::Unsupported(format!(
                    "JPEG process SOF{} (lossless, hierarchical or arithmetic) has no CPNP-1 decoder here",
                    marker - 0xC0
                )));
            }
            0xC4 => parse_dht(seg, &mut dc, &mut ac)?,
            0xDB => parse_dqt(seg, &mut qt)?,
            0xDD => {
                restart_interval = be16(seg, 0).ok_or_else(|| Error::Malformed("bad DRI".into()))?;
            }
            0xE0 => {
                if seg.starts_with(b"JFIF\0") {
                    saw_jfif = true;
                }
            }
            0xEE => {
                if seg.starts_with(b"Adobe") && seg.len() >= 12 {
                    adobe = seg.get(11).copied();
                }
            }
            0xDC => return Err(Error::Unsupported("DNL marker".into())),
            0xDA => {
                let fr = frame.as_mut().ok_or_else(|| Error::Malformed("scan before frame".into()))?;
                let end = decode_scan(bytes, seg, seg_end, fr, &dc, &ac, restart_interval)?;
                pos = end;
                continue;
            }
            _ => {}
        }
        pos = seg_end;
    }
    let fr = frame.ok_or_else(|| Error::Malformed("no image in JPEG".into()))?;
    finish(fr, &qt, saw_jfif, adobe)
}

fn parse_sof(seg: &[u8], progressive: bool) -> Result<Frame, Error> {
    let p = *seg.first().ok_or_else(|| Error::Malformed("bad SOF".into()))?;
    if p != 8 {
        return Err(Error::Unsupported(format!("{p}-bit JPEG samples")));
    }
    let height = be16(seg, 1).ok_or_else(|| Error::Malformed("bad SOF".into()))?;
    let width = be16(seg, 3).ok_or_else(|| Error::Malformed("bad SOF".into()))?;
    let n = usize::from(*seg.get(5).ok_or_else(|| Error::Malformed("bad SOF".into()))?);
    if width == 0 || height == 0 {
        return Err(Error::Unsupported("JPEG with zero or DNL-deferred dimensions".into()));
    }
    if !(n == 1 || n == 3 || n == 4) {
        return Err(Error::Unsupported(format!("{n}-component JPEG")));
    }
    if (width as u64) * (height as u64) > MAX_PIXELS {
        return Err(Error::Unsupported(format!("image too large: {width}×{height}")));
    }
    let mut comps = Vec::with_capacity(n);
    for i in 0..n {
        let c = seg.get(6 + i * 3..9 + i * 3).ok_or_else(|| Error::Malformed("bad SOF".into()))?;
        let (h, v) = (usize::from(c[1] >> 4), usize::from(c[1] & 15));
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) || c[2] > 3 {
            return Err(Error::Malformed("bad component parameters".into()));
        }
        comps.push(Component {
            id: c[0],
            h,
            v,
            tq: usize::from(c[2]),
            bw: 0,
            bh: 0,
            real_bw: 0,
            real_bh: 0,
            dw: 0,
            dh: 0,
            coefs: Vec::new(),
            dc_table: 0,
            ac_table: 0,
        });
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    let mcux = width.div_ceil(8 * hmax);
    let mcuy = height.div_ceil(8 * vmax);
    for c in &mut comps {
        c.bw = mcux * c.h;
        c.bh = mcuy * c.v;
        c.dw = (width * c.h).div_ceil(hmax);
        c.dh = (height * c.v).div_ceil(vmax);
        c.real_bw = c.dw.div_ceil(8);
        c.real_bh = c.dh.div_ceil(8);
        c.coefs = vec![[0i16; 64]; c.bw * c.bh];
    }
    Ok(Frame { progressive, width, height, hmax, vmax, mcux, mcuy, comps })
}

fn parse_dht(seg: &[u8], dc: &mut [Option<Huffman>; 4], ac: &mut [Option<Huffman>; 4]) -> Result<(), Error> {
    let mut p = 0usize;
    while p < seg.len() {
        let tc_th = *seg.get(p).ok_or_else(|| Error::Malformed("bad DHT".into()))?;
        let mut counts = [0u8; 16];
        counts.copy_from_slice(seg.get(p + 1..p + 17).ok_or_else(|| Error::Malformed("bad DHT".into()))?);
        let total: usize = counts.iter().map(|&c| usize::from(c)).sum();
        let values = seg.get(p + 17..p + 17 + total).ok_or_else(|| Error::Malformed("bad DHT".into()))?.to_vec();
        let (class, id) = (tc_th >> 4, usize::from(tc_th & 15));
        if id > 3 || class > 1 {
            return Err(Error::Malformed("bad DHT table id".into()));
        }
        let table = Huffman::build(&counts, values, class == 0)?;
        if class == 0 {
            dc[id] = Some(table);
        } else {
            ac[id] = Some(table);
        }
        p += 17 + total;
    }
    Ok(())
}

fn parse_dqt(seg: &[u8], qt: &mut [Option<[u16; 64]>; 4]) -> Result<(), Error> {
    let mut p = 0usize;
    while p < seg.len() {
        let pq_tq = *seg.get(p).ok_or_else(|| Error::Malformed("bad DQT".into()))?;
        let (prec, id) = (pq_tq >> 4, usize::from(pq_tq & 15));
        if id > 3 || prec > 1 {
            return Err(Error::Malformed("bad DQT".into()));
        }
        let mut t = [0u16; 64];
        for (k, slot) in NATURAL.iter().take(64).enumerate() {
            let v = if prec == 0 {
                u16::from(*seg.get(p + 1 + k).ok_or_else(|| Error::Malformed("bad DQT".into()))?)
            } else {
                be16(seg, p + 1 + 2 * k).ok_or_else(|| Error::Malformed("bad DQT".into()))? as u16
            };
            t[*slot] = v;
        }
        qt[id] = Some(t);
        p += 1 + if prec == 0 { 64 } else { 128 };
    }
    Ok(())
}

struct Scan {
    comps: Vec<usize>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    bytes: &[u8],
    seg: &[u8],
    data_start: usize,
    fr: &mut Frame,
    dc: &[Option<Huffman>; 4],
    ac: &[Option<Huffman>; 4],
    restart_interval: usize,
) -> Result<usize, Error> {
    let ns = usize::from(*seg.first().ok_or_else(|| Error::Malformed("bad SOS".into()))?);
    if ns == 0 || ns > 4 {
        return Err(Error::Malformed("bad SOS".into()));
    }
    let mut comps = Vec::with_capacity(ns);
    for i in 0..ns {
        let cs = *seg.get(1 + 2 * i).ok_or_else(|| Error::Malformed("bad SOS".into()))?;
        let t = *seg.get(2 + 2 * i).ok_or_else(|| Error::Malformed("bad SOS".into()))?;
        let ci = fr.comps.iter().position(|c| c.id == cs).ok_or_else(|| Error::Malformed("SOS names an unknown component".into()))?;
        if let Some(c) = fr.comps.get_mut(ci) {
            c.dc_table = usize::from(t >> 4).min(3);
            c.ac_table = usize::from(t & 15).min(3);
        }
        comps.push(ci);
    }
    let tail = seg.get(1 + 2 * ns..1 + 2 * ns + 3).ok_or_else(|| Error::Malformed("bad SOS".into()))?;
    let scan = Scan { comps, ss: usize::from(tail[0]), se: usize::from(tail[1]), ah: u32::from(tail[2] >> 4), al: u32::from(tail[2] & 15) };
    if fr.progressive {
        let dc_scan = scan.ss == 0;
        let bad = if dc_scan { scan.se != 0 } else { scan.se < scan.ss || scan.se > 63 || scan.comps.len() != 1 };
        if bad || scan.al > 13 || scan.ah > 13 {
            return Err(Error::Malformed("bad progression parameters".into()));
        }
    } else if scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0 {
        // libjpeg only warns for sequential files; it decodes the full block regardless.
    }
    // Tables needed by this scan must exist.
    for &ci in &scan.comps {
        let c = fr.comps.get(ci).ok_or_else(|| Error::Malformed("bad component".into()))?;
        let need_dc = !fr.progressive || (scan.ss == 0 && scan.ah == 0);
        let need_ac = !fr.progressive || scan.ss > 0;
        if need_dc && dc.get(c.dc_table).and_then(Option::as_ref).is_none() {
            return Err(Error::Malformed("missing DC Huffman table".into()));
        }
        if need_ac && ac.get(c.ac_table).and_then(Option::as_ref).is_none() {
            return Err(Error::Malformed("missing AC Huffman table".into()));
        }
    }

    let mut bits = Bits::new(bytes, data_start);
    let single = scan.comps.len() == 1;
    // MCU geometry: non-interleaved scans walk the component's real blocks one at a time.
    let (mcus_x, mcus_y) = if single {
        let c = fr.comps.get(scan.comps[0]).ok_or_else(|| Error::Malformed("bad component".into()))?;
        (c.real_bw, c.real_bh)
    } else {
        (fr.mcux, fr.mcuy)
    };
    let total = mcus_x * mcus_y;
    let mut last_dc = [0i32; 4];
    let mut eobrun = 0u32;
    let mut until_restart = restart_interval;
    for m in 0..total {
        if restart_interval > 0 {
            if until_restart == 0 {
                bits.restart();
                last_dc = [0; 4];
                eobrun = 0;
                until_restart = restart_interval;
            }
            until_restart -= 1;
        }
        let (mx, my) = (m % mcus_x, m / mcus_x);
        // libjpeg leaves an MCU at zero once the segment's data is exhausted.
        let skip = bits.insufficient;
        for (slot, &ci) in scan.comps.iter().enumerate() {
            let Some(c) = fr.comps.get_mut(ci) else { continue };
            let (bh, bv) = if single { (1, 1) } else { (c.h, c.v) };
            for by in 0..bv {
                for bx in 0..bh {
                    let (x, y) = if single { (mx, my) } else { (mx * c.h + bx, my * c.v + by) };
                    let idx = y * c.bw + x;
                    let Some(block) = c.coefs.get_mut(idx) else { continue };
                    if skip {
                        continue;
                    }
                    let dct = dc.get(c.dc_table).and_then(Option::as_ref);
                    let act = ac.get(c.ac_table).and_then(Option::as_ref);
                    let ld = last_dc.get_mut(slot).ok_or_else(|| Error::Malformed("bad scan".into()))?;
                    if !fr.progressive {
                        decode_block_sequential(&mut bits, block, dct, act, ld)?;
                    } else if scan.ss == 0 {
                        if scan.ah == 0 {
                            let s = match dct {
                                Some(t) => bits.decode(t)?,
                                None => 0,
                            };
                            let s = u32::from(s);
                            let r = bits.bits(s)?;
                            *ld = ld.wrapping_add(extend(r, s));
                            block[0] = (ld.wrapping_shl(scan.al)) as i16;
                        } else if bits.bit()? != 0 {
                            block[0] |= (1i32 << scan.al) as i16;
                        }
                    } else if scan.ah == 0 {
                        if let Some(t) = act {
                            decode_ac_first(&mut bits, block, t, &scan, &mut eobrun)?;
                        }
                    } else if let Some(t) = act {
                        decode_ac_refine(&mut bits, block, t, &scan, &mut eobrun)?;
                    }
                }
            }
        }
    }
    // Position after the scan: the next marker.
    let mut p = bits.pos;
    loop {
        match bytes.get(p) {
            None => return Ok(p),
            Some(0xFF) => match bytes.get(p + 1) {
                Some(0) => p += 2,
                Some(0xFF) => p += 1,
                Some(m) if (0xD0..=0xD7).contains(m) => p += 2,
                _ => return Ok(p),
            },
            Some(_) => p += 1,
        }
    }
}

fn decode_block_sequential(
    bits: &mut Bits,
    block: &mut [i16; 64],
    dc: Option<&Huffman>,
    ac: Option<&Huffman>,
    last_dc: &mut i32,
) -> Result<(), Error> {
    let s = match dc {
        Some(t) => u32::from(bits.decode(t)?),
        None => 0,
    };
    let r = bits.bits(s)?;
    *last_dc = last_dc.wrapping_add(extend(r, s));
    block[0] = *last_dc as i16;
    let Some(ac) = ac else { return Ok(()) };
    let mut k = 1usize;
    while k < 64 {
        let rs = bits.decode(ac)?;
        let (r, s) = (usize::from(rs >> 4), u32::from(rs & 15));
        if s != 0 {
            k += r;
            let v = extend(bits.bits(s)?, s);
            if let Some(slot) = block.get_mut(natural(k)) {
                *slot = v as i16;
            }
        } else {
            if r != 15 {
                break;
            }
            k += 15;
        }
        k += 1;
    }
    Ok(())
}

fn decode_ac_first(bits: &mut Bits, block: &mut [i16; 64], ac: &Huffman, scan: &Scan, eobrun: &mut u32) -> Result<(), Error> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = scan.ss;
    while k <= scan.se {
        let rs = bits.decode(ac)?;
        let (r, s) = (u32::from(rs >> 4), u32::from(rs & 15));
        if s != 0 {
            k += r as usize;
            let v = extend(bits.bits(s)?, s);
            if let Some(slot) = block.get_mut(natural(k)) {
                *slot = (v.wrapping_shl(scan.al)) as i16;
            }
        } else if r == 15 {
            k += 15;
        } else {
            *eobrun = 1 << r;
            if r != 0 {
                *eobrun += bits.bits(r)? as u32;
            }
            *eobrun -= 1;
            break;
        }
        k += 1;
    }
    Ok(())
}

fn refine(bits: &mut Bits, coef: &mut i16, p1: i32, m1: i32) -> Result<(), Error> {
    if bits.bit()? != 0 && (i32::from(*coef) & p1) == 0 {
        let v = if *coef >= 0 { i32::from(*coef) + p1 } else { i32::from(*coef) + m1 };
        *coef = v as i16;
    }
    Ok(())
}

fn decode_ac_refine(bits: &mut Bits, block: &mut [i16; 64], ac: &Huffman, scan: &Scan, eobrun: &mut u32) -> Result<(), Error> {
    let p1 = 1i32 << scan.al;
    let m1 = (-1i32).wrapping_shl(scan.al);
    let mut k = scan.ss;
    if *eobrun == 0 {
        while k <= scan.se {
            let rs = bits.decode(ac)?;
            let mut r = i32::from(rs >> 4);
            let s = rs & 15;
            let mut val = 0i32;
            if s != 0 {
                val = if bits.bit()? != 0 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = 1 << r;
                if r != 0 {
                    *eobrun += bits.bits(r as u32)? as u32;
                }
                break;
            }
            loop {
                let pos = natural(k);
                let nz = block.get(pos).is_some_and(|c| *c != 0);
                if nz {
                    if let Some(c) = block.get_mut(pos) {
                        refine(bits, c, p1, m1)?;
                    }
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > scan.se {
                    break;
                }
            }
            if val != 0
                && let Some(c) = block.get_mut(natural(k))
            {
                *c = val as i16;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= scan.se {
            let pos = natural(k);
            if block.get(pos).is_some_and(|c| *c != 0)
                && let Some(c) = block.get_mut(pos)
            {
                refine(bits, c, p1, m1)?;
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Reconstruction: islow IDCT, fancy upsampling, colour conversion
// ---------------------------------------------------------------------------------------------

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x.wrapping_add(1 << (n - 1))) >> n
}

/// libjpeg's `range_limit[x & RANGE_MASK]` for the IDCT output: the sample plus 128, clamped,
/// with the table's wrap-around for wildly out-of-range values.
#[inline]
fn idct_range_limit(x: i32) -> u8 {
    let i = x & 1023;
    if i < 128 {
        (i + 128) as u8
    } else if i < 512 {
        255
    } else if i < 896 {
        0
    } else {
        (i - 896) as u8
    }
}

/// The even/odd butterfly shared by both passes: returns the eight outputs before descaling.
#[inline]
fn idct_1d(z: [i32; 8]) -> [i32; 8] {
    // Even part.
    let (z2, z3) = (z[2], z[6]);
    let z1 = (z2.wrapping_add(z3)).wrapping_mul(FIX_0_541196100);
    let tmp2 = z1.wrapping_add(z3.wrapping_mul(-FIX_1_847759065));
    let tmp3 = z1.wrapping_add(z2.wrapping_mul(FIX_0_765366865));
    let (z2, z3) = (z[0], z[4]);
    let tmp0 = (z2.wrapping_add(z3)).wrapping_shl(CONST_BITS as u32);
    let tmp1 = (z2.wrapping_sub(z3)).wrapping_shl(CONST_BITS as u32);
    let tmp10 = tmp0.wrapping_add(tmp3);
    let tmp13 = tmp0.wrapping_sub(tmp3);
    let tmp11 = tmp1.wrapping_add(tmp2);
    let tmp12 = tmp1.wrapping_sub(tmp2);
    // Odd part.
    let (mut t0, mut t1, mut t2, mut t3) = (z[7], z[5], z[3], z[1]);
    let z1 = t0.wrapping_add(t3);
    let z2 = t1.wrapping_add(t2);
    let z3 = t0.wrapping_add(t2);
    let z4 = t1.wrapping_add(t3);
    let z5 = (z3.wrapping_add(z4)).wrapping_mul(FIX_1_175875602);
    t0 = t0.wrapping_mul(FIX_0_298631336);
    t1 = t1.wrapping_mul(FIX_2_053119869);
    t2 = t2.wrapping_mul(FIX_3_072711026);
    t3 = t3.wrapping_mul(FIX_1_501321110);
    let z1 = z1.wrapping_mul(-FIX_0_899976223);
    let z2 = z2.wrapping_mul(-FIX_2_562915447);
    let z3 = z3.wrapping_mul(-FIX_1_961570560).wrapping_add(z5);
    let z4 = z4.wrapping_mul(-FIX_0_390180644).wrapping_add(z5);
    t0 = t0.wrapping_add(z1).wrapping_add(z3);
    t1 = t1.wrapping_add(z2).wrapping_add(z4);
    t2 = t2.wrapping_add(z2).wrapping_add(z3);
    t3 = t3.wrapping_add(z1).wrapping_add(z4);
    [
        tmp10.wrapping_add(t3),
        tmp11.wrapping_add(t2),
        tmp12.wrapping_add(t1),
        tmp13.wrapping_add(t0),
        tmp13.wrapping_sub(t0),
        tmp12.wrapping_sub(t1),
        tmp11.wrapping_sub(t2),
        tmp10.wrapping_sub(t3),
    ]
}

/// `jpeg_idct_islow`: dequantise and inverse-transform one block into `out` (stride `stride`).
fn idct_islow(coef: &[i16; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for col in 0..8 {
        let deq = |r: usize| i32::from(coef[r * 8 + col]).wrapping_mul(i32::from(q[r * 8 + col]));
        if (1..8).all(|r| coef[r * 8 + col] == 0) {
            let dcval = deq(0).wrapping_shl(PASS1_BITS as u32);
            for r in 0..8 {
                ws[r * 8 + col] = dcval;
            }
            continue;
        }
        let z = [deq(0), deq(1), deq(2), deq(3), deq(4), deq(5), deq(6), deq(7)];
        let o = idct_1d(z);
        for r in 0..8 {
            ws[r * 8 + col] = descale(o[r], CONST_BITS - PASS1_BITS);
        }
    }
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let Some(dst) = out.get_mut(row * stride..row * stride + 8) else { continue };
        if w[1..].iter().all(|&v| v == 0) {
            let v = idct_range_limit(descale(w[0], PASS1_BITS + 3));
            dst.fill(v);
            continue;
        }
        let o = idct_1d([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        for (d, v) in dst.iter_mut().zip(o) {
            *d = idct_range_limit(descale(v, CONST_BITS + PASS1_BITS + 3));
        }
    }
}

/// One component's samples, `dw × dh` (no padding).
struct Plane {
    w: usize,
    h: usize,
    data: Vec<u8>,
}

impl Plane {
    #[inline]
    fn at(&self, x: usize, y: usize) -> i32 {
        let x = x.min(self.w.saturating_sub(1));
        let y = y.min(self.h.saturating_sub(1));
        i32::from(self.data.get(y * self.w + x).copied().unwrap_or(0))
    }
}

fn reconstruct(c: &Component, q: &[u16; 64]) -> Plane {
    let (pw, ph) = (c.bw * 8, c.bh * 8);
    let mut full = vec![0u8; pw * ph];
    for by in 0..c.bh {
        for bx in 0..c.bw {
            if let Some(block) = c.coefs.get(by * c.bw + bx)
                && let Some(dst) = full.get_mut(by * 8 * pw + bx * 8..)
            {
                idct_islow(block, q, dst, pw);
            }
        }
    }
    let mut data = Vec::with_capacity(c.dw * c.dh);
    for y in 0..c.dh {
        if let Some(row) = full.get(y * pw..y * pw + c.dw) {
            data.extend_from_slice(row);
        }
    }
    Plane { w: c.dw, h: c.dh, data }
}

/// Upsample `p` to `width × height` by libjpeg-turbo's rules for the ratio `(rx, ry)`.
fn upsample(p: &Plane, rx: usize, ry: usize, width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    match (rx, ry) {
        (1, 1) => {
            for y in 0..height {
                for x in 0..width {
                    out[y * width + x] = p.at(x, y) as u8;
                }
            }
        }
        (2, 1) => {
            // h2v1_fancy_upsample: 3/4 nearer + 1/4 further, alternating rounding bias.
            for y in 0..height {
                for x in 0..width {
                    let i = x / 2;
                    let v = p.at(i, y) * 3;
                    out[y * width + x] =
                        if x % 2 == 0 { ((v + p.at(i.saturating_sub(1), y) + 1) >> 2) as u8 } else { ((v + p.at(i + 1, y) + 2) >> 2) as u8 };
                }
            }
        }
        (2, 2) => {
            // h2v2_fancy_upsample: vertical 3:1 column sums, then horizontal 3:1 with biases 8/7.
            for y in 0..height {
                let r = y / 2;
                let far = if y % 2 == 0 { r.saturating_sub(1) } else { r + 1 };
                let colsum = |i: usize| p.at(i, r) * 3 + p.at(i, far);
                for x in 0..width {
                    let i = x / 2;
                    let this = colsum(i);
                    out[y * width + x] = if x % 2 == 0 {
                        ((this * 3 + colsum(i.saturating_sub(1)) + 8) >> 4) as u8
                    } else {
                        ((this * 3 + colsum(i + 1) + 7) >> 4) as u8
                    };
                }
            }
        }
        (1, 2) => {
            // h1v2_fancy_upsample (libjpeg-turbo): biases 1 (upper) and 2 (lower).
            for y in 0..height {
                let r = y / 2;
                let (far, bias) = if y % 2 == 0 { (r.saturating_sub(1), 1) } else { (r + 1, 2) };
                for x in 0..width {
                    out[y * width + x] = ((p.at(x, r) * 3 + p.at(x, far) + bias) >> 2) as u8;
                }
            }
        }
        _ => {
            // int_upsample: pixel replication.
            for y in 0..height {
                for x in 0..width {
                    out[y * width + x] = p.at(x / rx.max(1), y / ry.max(1)) as u8;
                }
            }
        }
    }
    out
}

/// `jdcolor.c`'s fixed-point YCbCr → RGB tables.
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

fn fix(x: f64) -> i32 {
    (x * f64::from(1 << SCALEBITS) + 0.5) as i32
}

impl YccTables {
    fn new() -> YccTables {
        let mut t = YccTables { cr_r: [0; 256], cb_b: [0; 256], cr_g: [0; 256], cb_g: [0; 256] };
        for i in 0..256usize {
            let x = i as i32 - 128;
            t.cr_r[i] = (fix(1.40200) * x + ONE_HALF) >> SCALEBITS;
            t.cb_b[i] = (fix(1.77200) * x + ONE_HALF) >> SCALEBITS;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + ONE_HALF;
        }
        t
    }

    #[inline]
    fn rgb(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let y = i32::from(y);
        let (cb, cr) = (usize::from(cb), usize::from(cr));
        let clamp = |v: i32| v.clamp(0, 255) as u8;
        [clamp(y + self.cr_r[cr]), clamp(y + ((self.cb_g[cb] + self.cr_g[cr]) >> SCALEBITS)), clamp(y + self.cb_b[cb])]
    }
}

fn finish(fr: Frame, qt: &[Option<[u16; 64]>; 4], saw_jfif: bool, adobe: Option<u8>) -> Result<Image, Error> {
    let (w, h) = (fr.width, fr.height);
    let mut planes = Vec::with_capacity(fr.comps.len());
    for c in &fr.comps {
        let q = qt.get(c.tq).and_then(|q| q.as_ref()).ok_or_else(|| Error::Malformed("missing quantisation table".into()))?;
        let p = reconstruct(c, q);
        let (rx, ry) = (fr.hmax / c.h, fr.vmax / c.v);
        let exact = fr.hmax.is_multiple_of(c.h) && fr.vmax.is_multiple_of(c.v);
        if !exact {
            return Err(Error::Unsupported("fractional chroma sampling ratio".into()));
        }
        planes.push(upsample(&p, rx, ry, w, h));
    }
    let n = w * h;
    let samples = match planes.len() {
        1 => Samples::Gray(planes.into_iter().next().unwrap_or_default()),
        3 => {
            // jdapimin.c default_decompress_parms: JFIF ⇒ YCbCr; Adobe transform 0 ⇒ RGB; else by ids.
            let ids: Vec<u8> = fr.comps.iter().map(|c| c.id).collect();
            let is_rgb = if saw_jfif {
                false
            } else if let Some(t) = adobe {
                t == 0
            } else {
                ids == *b"RGB"
            };
            let (p0, p1, p2) = (&planes[0], &planes[1], &planes[2]);
            let mut out = Vec::with_capacity(n * 3);
            if is_rgb {
                for i in 0..n {
                    out.extend_from_slice(&[p0[i], p1[i], p2[i]]);
                }
            } else {
                let t = YccTables::new();
                for i in 0..n {
                    out.extend_from_slice(&t.rgb(p0[i], p1[i], p2[i]));
                }
            }
            Samples::Rgb(out)
        }
        4 => {
            let ycck = adobe == Some(2);
            let (p0, p1, p2, p3) = (&planes[0], &planes[1], &planes[2], &planes[3]);
            let mut out = Vec::with_capacity(n * 4);
            if ycck {
                let t = YccTables::new();
                for i in 0..n {
                    let [r, g, b] = t.rgb(p0[i], p1[i], p2[i]);
                    out.extend_from_slice(&[255 - r, 255 - g, 255 - b, p3[i]]);
                }
            } else {
                for i in 0..n {
                    out.extend_from_slice(&[p0[i], p1[i], p2[i], p3[i]]);
                }
            }
            Samples::Cmyk(out)
        }
        k => return Err(Error::Unsupported(format!("{k}-component JPEG"))),
    };
    Ok(Image { width: w as u32, height: h as u32, samples })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_limit_table() {
        assert_eq!(idct_range_limit(0), 128);
        assert_eq!(idct_range_limit(127), 255);
        assert_eq!(idct_range_limit(-128), 0);
        assert_eq!(idct_range_limit(-1), 127);
        assert_eq!(idct_range_limit(400), 255);
        assert_eq!(idct_range_limit(-400), 0);
    }

    #[test]
    fn flat_block_is_flat() {
        let mut c = [0i16; 64];
        c[0] = 10;
        let q = [8u16; 64];
        let mut out = [0u8; 64];
        idct_islow(&c, &q, &mut out, 8);
        // 10·8 = 80 → DC value 80/8 = 10 → 138.
        assert!(out.iter().all(|&v| v == 138));
    }

    #[test]
    fn ycc_matches_libjpeg_constants() {
        let t = YccTables::new();
        assert_eq!(t.rgb(128, 128, 128), [128, 128, 128]);
        assert_eq!(t.rgb(0, 128, 255), [178, 0, 0]);
    }
}
