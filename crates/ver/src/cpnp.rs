//! CPNP-1: bytes in, one Canonical Buffer out.
//!
//! Executed as the published implementation profile `verd-pillow/1` (`standard/decoder-corpus/1/
//! PROFILE.md` §3) pins it, step by step:
//!
//! 1. **Decode** frame 0 at full resolution. JPEG goes through [`crate::jpeg`], a decoder that
//!    reproduces libjpeg-turbo's islow IDCT, fancy upsampling and colour tables; PNG through
//!    `png`, WebP through `image-webp`, TIFF through `tiff`, GIF through `gif` (coalesced onto
//!    the logical screen over a transparent canvas, background colour and disposal ignored,
//!    errata E7), BMP through `image`. AVIF, HEIC and camera raws have no CPNP-1 path here.
//! 2. **Orientation** from EXIF/TIFF IFD0 tag 274 only (E6; never XMP), by the explicit map.
//!    Out of range, wrong type or unreadable is upright.
//! 3. **Alpha** is set aside (a real alpha channel, or palette / colour-key transparency).
//! 4. **Colour**: an embedded ICC profile is converted to sRGB (relative colorimetric) through
//!    `moxcms`; a present-but-unusable profile fails closed (E8); no profile is sRGB.
//! 5. **Composite** over opaque white in the gamma domain with Pillow's `alpha_composite`
//!    integer arithmetic.
//! 6. **Quantise** to 8 bits: deeper sources are scaled, `round_half_up(v·255/max)` (E3).
//! 7. Never resize or crop.
//!
//! Where the profile leaves room (a case no corpus entry pins and the reference implementation
//! and an errata item disagree), the errata item wins and `docs/ver-cpnp.md` says so.

use crate::jpeg;
use crate::tiffifd::{self, Ifd};

/// A container format CPNP-1 knows (the schema's `image.format` enum).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Jpeg,
    Png,
    Webp,
    Avif,
    Heic,
    Tiff,
    Bmp,
    Gif,
}

impl Format {
    /// The schema's lowercase `image.format` token.
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Jpeg => "jpeg",
            Format::Png => "png",
            Format::Webp => "webp",
            Format::Avif => "avif",
            Format::Heic => "heic",
            Format::Tiff => "tiff",
            Format::Bmp => "bmp",
            Format::Gif => "gif",
        }
    }

    /// The format for a schema token.
    pub fn from_token(s: &str) -> Option<Format> {
        Some(match s {
            "jpeg" => Format::Jpeg,
            "png" => Format::Png,
            "webp" => Format::Webp,
            "avif" => Format::Avif,
            "heic" => Format::Heic,
            "tiff" => Format::Tiff,
            "bmp" => Format::Bmp,
            "gif" => Format::Gif,
            _ => return None,
        })
    }
}

/// The Canonical Buffer: row-major interleaved RGB8, `rgb.len() == width * height * 3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalBuffer {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub format: Format,
    pub animated: bool,
    pub alpha_present: bool,
    pub hdr_present: bool,
    pub icc_was_present: bool,
}

/// Why no Canonical Buffer exists for an input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpnpError {
    /// Not an image, or a damaged one.
    Undecodable(String),
    /// An embedded ICC profile is present but no transform to sRGB can be built (errata E8).
    UnusableIcc(String),
    /// A format with no CPNP-1 path here (camera raw, AVIF, HEIC, …).
    Unsupported(String),
}

impl std::fmt::Display for CpnpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpnpError::Undecodable(m) => write!(f, "undecodable: {m}"),
            CpnpError::UnusableIcc(m) => write!(f, "unusable ICC profile: {m}"),
            CpnpError::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for CpnpError {}

/// Refuse canvases larger than this (the reference has Pillow's decompression-bomb limit).
const MAX_PIXELS: u64 = 1 << 28;

/// The most a third-party decoder may allocate for one image (16-bit RGBA at [`MAX_PIXELS`] fits).
const DECODE_BUDGET: usize = 1 << 31;

/// The CPNP-1 format of `bytes` by signature, `None` for camera raws and unknowns.
pub fn is_cpnp_format(bytes: &[u8]) -> Option<Format> {
    use lightcraft_codecs::Format as F;
    match lightcraft_codecs::sniff(bytes)? {
        F::Jpeg => Some(Format::Jpeg),
        F::Png => Some(Format::Png),
        F::WebP => Some(Format::Webp),
        F::Avif => Some(Format::Avif),
        F::Heif => Some(Format::Heic),
        F::Tiff => Some(Format::Tiff),
        F::Bmp => Some(Format::Bmp),
        F::Gif => Some(Format::Gif),
        F::Jxl | F::Psd | F::RawTiffLike | F::RawOther => None,
    }
}

/// Colour samples before colour management.
#[derive(Clone, Debug)]
enum Px {
    Gray(Vec<u8>),
    Rgb(Vec<u8>),
    Cmyk(Vec<u8>),
}

impl Px {
    /// Pillow's conversion to RGB with no colour management (`L` replicates; `CMYK` uses
    /// `cmyk2rgb`).
    fn into_rgb(self) -> Vec<u8> {
        match self {
            Px::Rgb(v) => v,
            Px::Gray(v) => v.iter().flat_map(|&g| [g, g, g]).collect(),
            Px::Cmyk(v) => {
                let muldiv255 = |a: u32, b: u32| {
                    let t = a * b + 128;
                    ((t >> 8) + t) >> 8
                };
                v.chunks_exact(4)
                    .flat_map(|p| {
                        let nk = 255 - u32::from(p[3]);
                        let f = |c: u8| nk.saturating_sub(muldiv255(u32::from(c), nk)).min(255) as u8;
                        [f(p[0]), f(p[1]), f(p[2])]
                    })
                    .collect()
            }
        }
    }
}

/// A decoded frame 0 before CPNP-1's later steps.
struct Decoded {
    format: Format,
    w: usize,
    h: usize,
    px: Px,
    alpha: Option<Vec<u8>>,
    /// Embedded ICC profile (`Some(empty)` is treated as absent, like the reference's `bool()`).
    icc: Option<Vec<u8>>,
    orientation: u16,
    animated: bool,
}

fn undecodable(m: impl Into<String>) -> CpnpError {
    CpnpError::Undecodable(m.into())
}

fn check_size(w: usize, h: usize) -> Result<(), CpnpError> {
    if w == 0 || h == 0 {
        return Err(undecodable("image has no pixels"));
    }
    if (w as u64).saturating_mul(h as u64) > MAX_PIXELS {
        return Err(CpnpError::Unsupported(format!("image too large for CPNP-1 here: {w}×{h}")));
    }
    Ok(())
}

/// Run CPNP-1 over `bytes`.
pub fn canonicalize(bytes: &[u8]) -> Result<CanonicalBuffer, CpnpError> {
    if bytes.is_empty() {
        return Err(undecodable("empty source: nothing to decode"));
    }
    let format = match is_cpnp_format(bytes) {
        Some(f) => f,
        None => {
            return Err(match lightcraft_codecs::sniff(bytes) {
                Some(f) if f.is_raw() => CpnpError::Unsupported("camera raw: CPNP-1 defines no raw development".into()),
                Some(f) => CpnpError::Unsupported(format!("{f:?} is not a VER image.format")),
                None => undecodable("not a recognised image format"),
            });
        }
    };
    // Third-party decoders are not ours to trust with hostile input: a panic is an error.
    let decoded =
        std::panic::catch_unwind(|| decode(bytes, format)).unwrap_or_else(|_| Err(undecodable(format!("{} decoder panicked", format.as_str()))))?;
    finish(bytes, decoded)
}

fn decode(bytes: &[u8], format: Format) -> Result<Decoded, CpnpError> {
    match format {
        Format::Jpeg => decode_jpeg(bytes),
        Format::Png => decode_png(bytes),
        Format::Gif => decode_gif(bytes),
        Format::Webp => decode_webp(bytes),
        Format::Tiff => decode_tiff(bytes),
        Format::Bmp => decode_bmp(bytes),
        Format::Avif => Err(CpnpError::Unsupported("AVIF: no pure-Rust AV1 decoder in this build".into())),
        Format::Heic => Err(CpnpError::Unsupported("HEIC: no HEVC decoder in this build".into())),
    }
}

fn finish(bytes: &[u8], d: Decoded) -> Result<CanonicalBuffer, CpnpError> {
    let Decoded { format, w, h, px, alpha, icc, orientation, animated } = d;
    check_size(w, h)?;
    let icc = icc.filter(|p| !p.is_empty());
    let icc_was_present = icc.is_some();

    // 2. Orientation.
    let (px, ow, oh) = match px {
        Px::Gray(v) => {
            let (v, a, b) = orient(&v, w, h, 1, orientation);
            (Px::Gray(v), a, b)
        }
        Px::Rgb(v) => {
            let (v, a, b) = orient(&v, w, h, 3, orientation);
            (Px::Rgb(v), a, b)
        }
        Px::Cmyk(v) => {
            let (v, a, b) = orient(&v, w, h, 4, orientation);
            (Px::Cmyk(v), a, b)
        }
    };
    let alpha = alpha.map(|a| orient(&a, w, h, 1, orientation).0);

    // 4. HDR: flagged from markers, never tone-mapped.
    let hdr_present = has_gain_map(bytes);

    // 5a. Alpha set aside; with alpha present the colour goes to RGB before colour management
    // (the reference's `img.convert("RGB")`).
    let alpha_present = alpha.is_some();
    let px = if alpha_present { Px::Rgb(px.into_rgb()) } else { px };

    // 3. Colour.
    let mut rgb = match &icc {
        Some(profile) => to_srgb(px, profile, ow, oh)?,
        None => px.into_rgb(),
    };

    // 5b. Composite over white.
    if let Some(a) = alpha {
        composite_over_white(&mut rgb, &a);
    }

    if rgb.len() != ow * oh * 3 {
        return Err(undecodable("internal: canonical buffer has the wrong size"));
    }
    Ok(CanonicalBuffer {
        width: u32::try_from(ow).map_err(|_| undecodable("width overflow"))?,
        height: u32::try_from(oh).map_err(|_| undecodable("height overflow"))?,
        rgb,
        format,
        animated,
        alpha_present,
        hdr_present,
        icc_was_present,
    })
}

/// Apply EXIF orientation `o` to a `w × h` buffer of `ch`-byte pixels.
fn orient(src: &[u8], w: usize, h: usize, ch: usize, o: u16) -> (Vec<u8>, usize, usize) {
    if !(2..=8).contains(&o) {
        return (src.to_vec(), w, h);
    }
    let transpose = o >= 5;
    let (ow, oh) = if transpose { (h, w) } else { (w, h) };
    let mut out = vec![0u8; src.len()];
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = match o {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                _ => (w - 1 - y, x),
            };
            let s = (sy * w + sx) * ch;
            let d = (y * ow + x) * ch;
            if let (Some(dst), Some(px)) = (out.get_mut(d..d + ch), src.get(s..s + ch)) {
                dst.copy_from_slice(px);
            }
        }
    }
    (out, ow, oh)
}

const GAIN_MAP_MARKERS: [&[u8]; 3] = [b"urn:iso:std:iso:ts:21496:-1", b"hdrgm:Version", b"GainMapImage"];

fn has_gain_map(bytes: &[u8]) -> bool {
    GAIN_MAP_MARKERS.iter().any(|m| bytes.windows(m.len()).any(|w| w == *m))
}

/// Pillow's `Image.alpha_composite(white, rgba)` per pixel (`AlphaComposite.c`, 7 precision bits).
pub(crate) fn composite_pixel(c: u8, a: u8) -> u8 {
    if a == 0 {
        return 255;
    }
    let a = u32::from(a);
    let blend = 255 * (255 - a);
    let outa255 = a * 255 + blend;
    let coef1 = a * 255 * 255 * (1 << 7) / outa255;
    let coef2 = 255 * (1 << 7) - coef1;
    let tmp = u32::from(c) * coef1 + 255 * coef2 + (0x80 << 7);
    ((((tmp >> 8) + tmp) >> 8) >> 7).min(255) as u8
}

fn composite_over_white(rgb: &mut [u8], alpha: &[u8]) {
    for (p, &a) in rgb.chunks_exact_mut(3).zip(alpha) {
        for c in p {
            *c = composite_pixel(*c, a);
        }
    }
}

/// 3. Convert to sRGB through the embedded profile (relative colorimetric). Fails closed (E8).
fn to_srgb(px: Px, icc: &[u8], w: usize, h: usize) -> Result<Vec<u8>, CpnpError> {
    use moxcms::{ColorProfile, Layout, RenderingIntent, TransformOptions};
    let unusable = |m: String| CpnpError::UnusableIcc(format!("{m}; no Canonical Buffer exists for this asset (errata E8)"));
    let guarded = |f: &dyn Fn() -> Result<Vec<u8>, CpnpError>| -> Result<Vec<u8>, CpnpError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .unwrap_or_else(|_| Err(unusable("the colour engine panicked on this profile".into())))
    };
    guarded(&|| {
        let src = ColorProfile::new_from_slice(icc).map_err(|e| unusable(format!("profile does not parse ({e:?})")))?;
        let dst = ColorProfile::new_srgb();
        let opts = TransformOptions { rendering_intent: RenderingIntent::RelativeColorimetric, ..TransformOptions::default() };
        let (layout, data) = match &px {
            Px::Gray(v) => (Layout::Gray, v),
            Px::Rgb(v) => (Layout::Rgb, v),
            Px::Cmyk(v) => (Layout::Rgba, v),
        };
        if let Px::Rgb(v) = &px
            && let Some(out) = crate::cms::rgb_matrix_shaper(&src, v)
        {
            return Ok(out);
        }
        if let Px::Gray(v) = &px
            && let Some(out) = crate::cms::gray_trc(&src, v)
        {
            return Ok(out);
        }
        // LUT-based, CMYK or Lab profiles: moxcms in f32, rounded once (its 8-bit path truncates).
        let t = src.create_transform_f32(layout, &dst, Layout::Rgb, opts).map_err(|e| unusable(format!("no transform to sRGB ({e:?})")))?;
        let input: Vec<f32> = data.iter().map(|&v| f32::from(v) / 255.0).collect();
        let mut fout = vec![0f32; w * h * 3];
        t.transform(&input, &mut fout).map_err(|e| unusable(format!("transform failed ({e:?})")))?;
        let out = fout.iter().map(|&v| if v.is_finite() { (v * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8 } else { 0 }).collect();
        Ok(out)
    })
}

// ---------------------------------------------------------------------------------------------
// Per-format decoding
// ---------------------------------------------------------------------------------------------

fn decode_jpeg(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    let img = jpeg::decode(bytes).map_err(|e| match e {
        jpeg::Error::Unsupported(m) => CpnpError::Unsupported(format!("JPEG: {m}")),
        jpeg::Error::Malformed(m) => undecodable(format!("JPEG: {m}")),
    })?;
    let meta = jpeg_meta(bytes);
    let px = match img.samples {
        jpeg::Samples::Gray(v) => Px::Gray(v),
        jpeg::Samples::Rgb(v) => Px::Rgb(v),
        // Pillow reads every CMYK JPEG with rawmode "CMYK;I" (Adobe's inverted convention).
        jpeg::Samples::Cmyk(v) => Px::Cmyk(v.into_iter().map(|c| 255 - c).collect()),
    };
    Ok(Decoded {
        format: Format::Jpeg,
        w: img.width as usize,
        h: img.height as usize,
        px,
        alpha: None,
        icc: meta.icc,
        orientation: meta.exif.as_deref().map(tiffifd::exif_orientation).unwrap_or(1),
        animated: false,
    })
}

struct JpegMeta {
    exif: Option<Vec<u8>>,
    icc: Option<Vec<u8>>,
}

/// The first `Exif\0\0` APP1 (Pillow keeps the first), and the APP2 ICC chunks reassembled the
/// way Pillow does: sorted by sequence number, used only when the count matches.
fn jpeg_meta(bytes: &[u8]) -> JpegMeta {
    let mut exif = None;
    let mut icc_chunks: Vec<&[u8]> = Vec::new();
    for (marker, payload) in crate::segments::jpeg_markers(bytes) {
        if marker == 0xE1 && exif.is_none() && payload.starts_with(b"Exif\0\0") {
            exif = Some(payload.to_vec());
        }
        if marker == 0xE2 && payload.starts_with(b"ICC_PROFILE\0") {
            icc_chunks.push(payload);
        }
    }
    let icc = if icc_chunks.is_empty() {
        None
    } else {
        icc_chunks.sort();
        let declared = icc_chunks.first().and_then(|c| c.get(13)).copied().map(usize::from);
        if declared == Some(icc_chunks.len()) {
            Some(icc_chunks.iter().flat_map(|c| c.get(14..).unwrap_or(&[]).iter().copied()).collect())
        } else {
            None
        }
    };
    JpegMeta { exif, icc }
}

/// Scale a `bits`-deep sample to 8 bits: `round_half_up(v·255/max)` (E3).
fn scale_to_8(v: u32, max: u32) -> u8 {
    if max == 255 {
        return v.min(255) as u8;
    }
    let max = u64::from(max.max(1));
    ((u64::from(v) * 255 * 2 + max) / (2 * max)).min(255) as u8
}

fn decode_png(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    let chunks = crate::segments::png_chunks(bytes).ok_or_else(|| undecodable("PNG: malformed chunk structure"))?;
    // ICC: iCCP, inflated here so a damaged stream fails closed instead of reading as absent.
    let mut icc = None;
    let mut exif = None;
    let mut raw_profile_exif = None;
    for (typ, data) in &chunks {
        match typ {
            b"iCCP" => {
                let nul = data.iter().position(|&b| b == 0).ok_or_else(|| CpnpError::UnusableIcc("iCCP chunk without a profile name".into()))?;
                let compressed = data.get(nul + 2..).ok_or_else(|| CpnpError::UnusableIcc("iCCP chunk truncated".into()))?;
                let p = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(compressed, 64 << 20).map_err(|_| {
                    CpnpError::UnusableIcc("iCCP profile does not inflate; no Canonical Buffer exists for this asset (errata E8)".into())
                })?;
                icc = Some(p);
            }
            b"eXIf" => exif = Some(data.to_vec()),
            b"tEXt" | b"zTXt" | b"iTXt" => {
                if raw_profile_exif.is_none()
                    && let Some(text) = png_text(typ, data, b"Raw profile type exif")
                {
                    raw_profile_exif = raw_profile_bytes(&text);
                }
            }
            _ => {}
        }
    }
    let orientation = exif.or(raw_profile_exif).as_deref().map(tiffifd::exif_orientation).unwrap_or(1);

    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    // The default 64 MiB refuses ordinary 24 MP photographs; the canvas cap above bounds it.
    dec.set_limits(png::Limits { bytes: DECODE_BUDGET });
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().map_err(|e| undecodable(format!("PNG: {e}")))?;
    let (w, h) = (reader.info().width as usize, reader.info().height as usize);
    check_size(w, h)?;
    let animated = reader.info().animation_control.as_ref().map(|a| {
        let extra = u32::from(reader.info().frame_control.is_none());
        a.num_frames.saturating_add(extra)
    });
    let size = reader.output_buffer_size().ok_or_else(|| undecodable("PNG: image too large"))?;
    let mut buf = vec![0u8; size];
    let out = reader.next_frame(&mut buf).map_err(|e| undecodable(format!("PNG: {e}")))?;
    let deep = out.bit_depth == png::BitDepth::Sixteen;
    let samples: Vec<u8> =
        if deep { buf.chunks_exact(2).map(|c| scale_to_8(u32::from(u16::from_be_bytes([c[0], c[1]])), 65535)).collect() } else { buf };
    let n = w * h;
    let (px, alpha) = split_channels(
        &samples,
        n,
        match out.color_type {
            png::ColorType::Grayscale => Layout::Gray,
            png::ColorType::GrayscaleAlpha => Layout::GrayA,
            png::ColorType::Rgb => Layout::Rgb,
            png::ColorType::Rgba => Layout::Rgba,
            png::ColorType::Indexed => return Err(undecodable("PNG: palette not expanded")),
        },
    )?;
    Ok(Decoded { format: Format::Png, w, h, px, alpha, icc, orientation, animated: animated.is_some_and(|n| n > 1) })
}

#[derive(Clone, Copy)]
enum Layout {
    Gray,
    GrayA,
    Rgb,
    Rgba,
}

fn split_channels(s: &[u8], n: usize, layout: Layout) -> Result<(Px, Option<Vec<u8>>), CpnpError> {
    let need = |c: usize| if s.len() >= n * c { Ok(()) } else { Err(undecodable("short pixel buffer")) };
    Ok(match layout {
        Layout::Gray => {
            need(1)?;
            (Px::Gray(s[..n].to_vec()), None)
        }
        Layout::GrayA => {
            need(2)?;
            let (g, a) = s[..n * 2].chunks_exact(2).map(|p| (p[0], p[1])).unzip();
            (Px::Gray(g), Some(a))
        }
        Layout::Rgb => {
            need(3)?;
            (Px::Rgb(s[..n * 3].to_vec()), None)
        }
        Layout::Rgba => {
            need(4)?;
            let mut rgb = Vec::with_capacity(n * 3);
            let mut a = Vec::with_capacity(n);
            for p in s[..n * 4].chunks_exact(4) {
                rgb.extend_from_slice(&p[..3]);
                a.push(p[3]);
            }
            (Px::Rgb(rgb), Some(a))
        }
    })
}

/// The text of a PNG text chunk whose keyword is `key` (inflating `zTXt` / compressed `iTXt`).
fn png_text(typ: &[u8; 4], data: &[u8], key: &[u8]) -> Option<Vec<u8>> {
    let nul = data.iter().position(|&b| b == 0)?;
    if data.get(..nul)? != key {
        return None;
    }
    let rest = data.get(nul + 1..)?;
    match typ {
        b"tEXt" => Some(rest.to_vec()),
        b"zTXt" => miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(rest.get(1..)?, 64 << 20).ok(),
        _ => {
            // iTXt: compression flag, method, language\0, translated keyword\0, text.
            let flag = *rest.first()?;
            let after = rest.get(2..)?;
            let l1 = after.iter().position(|&b| b == 0)?;
            let after = after.get(l1 + 1..)?;
            let l2 = after.iter().position(|&b| b == 0)?;
            let text = after.get(l2 + 1..)?;
            if flag == 1 { miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(text, 64 << 20).ok() } else { Some(text.to_vec()) }
        }
    }
}

/// ImageMagick's "Raw profile type exif" text: three header lines, then hex (as Pillow reads it).
fn raw_profile_bytes(text: &[u8]) -> Option<Vec<u8>> {
    let s = std::str::from_utf8(text).ok()?;
    let hex: String = s.split('\n').skip(3).collect::<Vec<_>>().join("");
    let hex: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    hex.chunks_exact(2).map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok()).collect()
}

fn decode_gif(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::Indexed);
    opts.check_frame_consistency(false);
    if let Some(limit) = std::num::NonZeroU64::new(DECODE_BUDGET as u64) {
        opts.set_memory_limit(gif::MemoryLimit::Bytes(limit));
    }
    let mut dec = opts.read_info(std::io::Cursor::new(bytes)).map_err(|e| undecodable(format!("GIF: {e}")))?;
    let (sw, sh) = (usize::from(dec.width()), usize::from(dec.height()));
    let global: Option<Vec<u8>> = dec.global_palette().map(<[u8]>::to_vec);
    let frame = dec.read_next_frame().map_err(|e| undecodable(format!("GIF: {e}")))?.ok_or_else(|| undecodable("GIF: no image"))?;
    let (fx, fy, fw, fh) = (usize::from(frame.left), usize::from(frame.top), usize::from(frame.width), usize::from(frame.height));
    // Pillow grows the canvas to hold a frame that overruns the logical screen.
    let (w, h) = (sw.max(fx + fw), sh.max(fy + fh));
    check_size(w, h)?;
    let palette = frame.palette.clone().or(global);
    let colour = |i: u8| -> [u8; 3] {
        match &palette {
            Some(p) => {
                let k = usize::from(i) * 3;
                match p.get(k..k + 3) {
                    Some(c) => [c[0], c[1], c[2]],
                    // Pillow's palette is a grey ramp beyond the entries the file supplies.
                    None => [i, i, i],
                }
            }
            None => [i, i, i],
        }
    };
    let transparent = frame.transparent;
    let covers = fx == 0 && fy == 0 && fw >= w && fh >= h;
    let mut rgb = vec![0u8; w * h * 3];
    // E7: the canvas under frame 0 is transparent.
    let mut alpha = vec![0u8; w * h];
    for y in 0..fh {
        for x in 0..fw {
            let Some(&i) = frame.buffer.get(y * fw + x) else { continue };
            let (cx, cy) = (fx + x, fy + y);
            let d = cy * w + cx;
            if transparent == Some(i) {
                continue;
            }
            if let (Some(px), Some(a)) = (rgb.get_mut(d * 3..d * 3 + 3), alpha.get_mut(d)) {
                px.copy_from_slice(&colour(i));
                *a = 255;
            }
        }
    }
    let frames = gif_frame_count(bytes);
    let has_alpha = transparent.is_some() || !covers;
    Ok(Decoded { format: Format::Gif, w, h, px: Px::Rgb(rgb), alpha: has_alpha.then_some(alpha), icc: None, orientation: 1, animated: frames > 1 })
}

/// Image descriptors in a GIF stream (walking blocks, tolerating truncation as Pillow's
/// `n_frames` does).
fn gif_frame_count(b: &[u8]) -> usize {
    let mut p = 13usize;
    let Some(&flags) = b.get(10) else { return 0 };
    if flags & 0x80 != 0 {
        p += 3 << ((flags & 7) + 1);
    }
    let skip_subblocks = |mut p: usize| -> Option<usize> {
        loop {
            let n = usize::from(*b.get(p)?);
            p += 1;
            if n == 0 {
                return Some(p);
            }
            p += n;
        }
    };
    let mut frames = 0;
    loop {
        match b.get(p) {
            Some(0x21) => match skip_subblocks(p + 2) {
                Some(q) => p = q,
                None => return frames,
            },
            Some(0x2C) => {
                let Some(&f) = b.get(p + 9) else { return frames };
                p += 10;
                if f & 0x80 != 0 {
                    p += 3 << ((f & 7) + 1);
                }
                frames += 1;
                match skip_subblocks(p + 1) {
                    Some(q) => p = q,
                    None => return frames,
                }
            }
            _ => return frames,
        }
    }
}

fn decode_webp(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    let map = |e: image_webp::DecodingError| undecodable(format!("WebP: {e}"));
    let mut dec = image_webp::WebPDecoder::new(std::io::Cursor::new(bytes)).map_err(map)?;
    let (w, h) = dec.dimensions();
    let (w, h) = (w as usize, h as usize);
    check_size(w, h)?;
    let icc = dec.icc_profile().map_err(map)?;
    let exif = dec.exif_metadata().map_err(map)?;
    let has_alpha = dec.has_alpha();
    let animated = dec.is_animated() && dec.num_frames() > 1;
    let size = dec.output_buffer_size().ok_or_else(|| undecodable("WebP: image too large"))?;
    let mut buf = vec![0u8; size];
    if dec.is_animated() {
        dec.read_frame(&mut buf).map_err(map)?;
    } else {
        dec.read_image(&mut buf).map_err(map)?;
    }
    let (px, alpha) = split_channels(&buf, w * h, if has_alpha { Layout::Rgba } else { Layout::Rgb })?;
    Ok(Decoded { format: Format::Webp, w, h, px, alpha, icc, orientation: exif.as_deref().map(tiffifd::exif_orientation).unwrap_or(1), animated })
}

fn decode_tiff(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    let ifd = Ifd::parse(bytes).ok_or_else(|| undecodable("TIFF: no IFD"))?;
    let orientation = tiffifd::orientation(&ifd);
    let icc = ifd.get(34675).map(|e| e.value.to_vec());
    let extra = ifd.uints(338).and_then(|v| v.first().copied());
    let colormap = ifd.uints(320);
    let photometric = ifd.uints(262).and_then(|v| v.first().copied());

    let map = |e: tiff::TiffError| undecodable(format!("TIFF: {e}"));
    let dec = Decoder::new(std::io::Cursor::new(bytes)).map_err(map)?;
    let mut limits = Limits::default();
    limits.decoding_buffer_size = 1 << 31;
    limits.intermediate_buffer_size = 1 << 30;
    let mut dec = dec.with_limits(limits);
    let (w, h) = dec.dimensions().map_err(map)?;
    let (w, h) = (w as usize, h as usize);
    check_size(w, h)?;
    let ct = dec.colortype().map_err(map)?;
    let img = dec.read_image().map_err(map)?;
    let s: Vec<u8> = match img {
        DecodingResult::U8(v) => v,
        DecodingResult::U16(v) => v.into_iter().map(|x| scale_to_8(u32::from(x), 65535)).collect(),
        DecodingResult::F32(v) => v.into_iter().map(|x| float_to_8(f64::from(x))).collect(),
        DecodingResult::F64(v) => v.into_iter().map(float_to_8).collect(),
        _ => return Err(CpnpError::Unsupported("TIFF: sample format".into())),
    };
    let n = w * h;
    use tiff::ColorType as T;
    let (px, alpha) = match ct {
        T::Gray(_) => {
            let (px, _) = split_channels(&s, n, Layout::Gray)?;
            // PhotometricInterpretation 0 (WhiteIsZero) is inverted by Pillow's "L;I" rawmode.
            match (photometric, px) {
                (Some(0), Px::Gray(g)) => (Px::Gray(g.into_iter().map(|v| 255 - v).collect()), None),
                (_, px) => (px, None),
            }
        }
        T::GrayA(_) => split_channels(&s, n, Layout::GrayA)?,
        T::RGB(_) => split_channels(&s, n, Layout::Rgb)?,
        T::RGBA(_) => {
            let (px, a) = split_channels(&s, n, Layout::Rgba)?;
            match extra {
                // Associated alpha: Pillow's "RGBa" → "RGBA" unpremultiplies (truncating).
                Some(1) => {
                    let a = a.unwrap_or_default();
                    let Px::Rgb(mut rgb) = px else { return Err(undecodable("TIFF: layout")) };
                    for (p, &al) in rgb.chunks_exact_mut(3).zip(&a) {
                        if al != 0 && al != 255 {
                            for c in p {
                                *c = (255 * u32::from(*c) / u32::from(al)).min(255) as u8;
                            }
                        }
                    }
                    (Px::Rgb(rgb), Some(a))
                }
                Some(2) => (px, a),
                // No or unspecified extra-sample meaning: Pillow reads "RGBX" and drops it.
                _ => (px, None),
            }
        }
        T::Palette(bits) => {
            let cmap = colormap.ok_or_else(|| undecodable("TIFF: palette image without ColorMap"))?;
            let entries = 1usize << bits.min(8);
            if cmap.len() < entries * 3 {
                return Err(undecodable("TIFF: short ColorMap"));
            }
            // Pillow keeps the high byte of each 16-bit ColorMap entry.
            let c8 = |v: u64| (v / 256).min(255) as u8;
            let mut rgb = Vec::with_capacity(n * 3);
            for &i in s.iter().take(n) {
                let i = usize::from(i).min(entries - 1);
                rgb.extend_from_slice(&[c8(cmap[i]), c8(cmap[entries + i]), c8(cmap[2 * entries + i])]);
            }
            (Px::Rgb(rgb), None)
        }
        T::CMYK(_) => {
            if s.len() < n * 4 {
                return Err(undecodable("short pixel buffer"));
            }
            (Px::Cmyk(s[..n * 4].to_vec()), None)
        }
        other => return Err(CpnpError::Unsupported(format!("TIFF colour type {other:?}"))),
    };
    Ok(Decoded { format: Format::Tiff, w, h, px, alpha, icc, orientation, animated: false })
}

fn float_to_8(v: f64) -> u8 {
    if !v.is_finite() {
        return if v > 0.0 { 255 } else { 0 };
    }
    // round_half_up(v·255), clamped (E3, normalised floating point).
    (v * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
}

fn decode_bmp(bytes: &[u8]) -> Result<Decoded, CpnpError> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Bmp).map_err(|e| undecodable(format!("BMP: {e}")))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    check_size(w, h)?;
    let (px, alpha) = if img.color().has_alpha() {
        split_channels(img.to_rgba8().as_raw(), w * h, Layout::Rgba)?
    } else {
        split_channels(img.to_rgb8().as_raw(), w * h, Layout::Rgb)?
    };
    Ok(Decoded { format: Format::Bmp, w, h, px, alpha, icc: None, orientation: 1, animated: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_extremes() {
        assert_eq!(composite_pixel(0, 0), 255);
        for c in 0..=255u8 {
            assert_eq!(composite_pixel(c, 255), c);
        }
    }

    #[test]
    fn scale_rounds_half_up() {
        assert_eq!(scale_to_8(65535, 65535), 255);
        assert_eq!(scale_to_8(0, 65535), 0);
        assert_eq!(scale_to_8(257, 65535), 1);
        assert_eq!(scale_to_8(128, 257), 127); // 127.0 exactly
    }

    #[test]
    fn orientation_map_dimensions() {
        let src: Vec<u8> = (0..6).collect(); // 3×2, one channel
        assert_eq!(orient(&src, 3, 2, 1, 1).0, src);
        assert_eq!(orient(&src, 3, 2, 1, 2).0, vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(orient(&src, 3, 2, 1, 3).0, vec![5, 4, 3, 2, 1, 0]);
        assert_eq!(orient(&src, 3, 2, 1, 4).0, vec![3, 4, 5, 0, 1, 2]);
        assert_eq!(orient(&src, 3, 2, 1, 5), (vec![0, 3, 1, 4, 2, 5], 2, 3));
        assert_eq!(orient(&src, 3, 2, 1, 6), (vec![3, 0, 4, 1, 5, 2], 2, 3));
        assert_eq!(orient(&src, 3, 2, 1, 7), (vec![5, 2, 4, 1, 3, 0], 2, 3));
        assert_eq!(orient(&src, 3, 2, 1, 8), (vec![2, 5, 1, 4, 0, 3], 2, 3));
        assert_eq!(orient(&src, 3, 2, 1, 99).0, src);
    }
}
