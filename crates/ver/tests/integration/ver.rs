//! lightcraft-ver behaviour: hashes, perceptual digests, segments, records, validation against
//! the vendored conformance fixtures, and never crashing.

use crate::common::*;
use lightcraft_ver::cpnp::{self, CanonicalBuffer, Format};
use lightcraft_ver::record::{self, Record, Schema};
use lightcraft_ver::segments::{self, State};
use lightcraft_ver::validate::{self, Severity};
use lightcraft_ver::{b64, hashing};
use serde_json::Value;
use sha2::Digest;

fn sha256(b: &[u8]) -> String {
    hashing::sha256_hex(b)
}

// ---------------------------------------------------------------------------------------------
// Vendored files
// ---------------------------------------------------------------------------------------------

#[test]
fn vendored_schemas_are_byte_identical_to_upstream() {
    // sha256 of standards/ver/1.0.1/ver-record.schema.json and standards/ver/1.1-draft/ver-record.schema.json.
    let s = standard().join("schemas");
    assert_eq!(
        sha256(&std::fs::read(s.join("ver-record-1.0.1.schema.json")).expect("read")),
        "44166d59f31e8bb898e0a3bf4c5863c154e273a97d1bc6bfee61d141e949d34a"
    );
    assert_eq!(
        sha256(&std::fs::read(s.join("ver-record-1.1-draft.schema.json")).expect("read")),
        "a424b4967cf39943c53a83df3869b674e42efb61aac7d23495b778c8189dd31e"
    );
    // The decoder corpus manifest's own digest, as its README records it.
    assert_eq!(
        sha256(&std::fs::read(standard().join("decoder-corpus/1/manifest.json")).expect("read")),
        "d743f077778d60b724f060355a26ea138c9f92ee1d82c275a76707a073e09b5e"
    );
}

// ---------------------------------------------------------------------------------------------
// Hashes
// ---------------------------------------------------------------------------------------------

#[test]
fn pixel_hash_preimage_by_hand() {
    let buf = CanonicalBuffer {
        width: 2,
        height: 1,
        rgb: vec![255, 0, 0, 0, 0, 255],
        format: Format::Png,
        animated: false,
        alpha_present: false,
        hdr_present: false,
        icc_was_present: false,
    };
    let mut pre = b"VER1:2:1:sRGB8:".to_vec();
    pre.extend_from_slice(&[255, 0, 0, 0, 0, 255]);
    let want = sha2::Sha256::digest(&pre).iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert_eq!(hashing::pixel_hash(&buf), want);
    assert_eq!(want.len(), 64);
    assert_eq!(hashing::content_hash(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

#[test]
fn perceptual_digests_have_registered_widths() {
    for name in ["jpeg-q75-420.jpg", "png-rgb.png", "gif-palette-transparency.gif", "webp-lossless.webp"] {
        let (_, id) = hashing::identity(&corpus_file(name)).expect("identity");
        assert_eq!(id.pdq.len(), 64, "{name}");
        assert_eq!(id.phash.len(), 16, "{name}");
        assert!(id.pdq.bytes().chain(id.phash.bytes()).all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
    }
}

fn encode_jpeg(buf: &CanonicalBuffer, quality: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut out, quality);
    enc.encode(&buf.rgb, buf.width as u16, buf.height as u16, jpeg_encoder::ColorType::Rgb).expect("encode");
    out
}

fn encode_png(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("header");
        w.write_image_data(rgb).expect("data");
    }
    out
}

fn hamming_hex(a: &str, b: &str) -> u32 {
    a.as_bytes()
        .chunks(2)
        .zip(b.as_bytes().chunks(2))
        .map(|(x, y)| {
            let p = u8::from_str_radix(std::str::from_utf8(x).expect("hex"), 16).expect("hex");
            let q = u8::from_str_radix(std::str::from_utf8(y).expect("hex"), 16).expect("hex");
            (p ^ q).count_ones()
        })
        .sum()
}

#[test]
fn pdq_survives_a_jpeg_reencode() {
    let png = corpus_file("png-rgb.png");
    let a = cpnp::canonicalize(&png).expect("png");
    let b = cpnp::canonicalize(&encode_jpeg(&a, 85)).expect("jpeg");
    let d = hamming_hex(&hashing::pdq(&a), &hashing::pdq(&b));
    assert!(d <= 16, "PDQ moved {d} bits under a q85 JPEG re-encode");
}

#[test]
fn perceptual_digests_are_over_the_full_canonical_buffer() {
    let (w, h) = (1200u32, 900u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            rgb.extend_from_slice(&[((x * 255) / w) as u8, ((y * 255) / h) as u8, (((x / 37) ^ (y / 23)) * 41 % 256) as u8]);
        }
    }
    let buf = cpnp::canonicalize(&encode_png(w, h, &rgb)).expect("png");
    assert_eq!((buf.width, buf.height), (w, h));
    assert_eq!(buf.rgb, rgb);
    let d = lightcraft_vera::digest_rgb8(w as usize, h as usize, &buf.rgb).expect("digest");
    let direct: String = d.pdq.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hashing::pdq(&buf), direct, "PDQ is computed over the full {w}×{h} buffer");
    assert_eq!(hashing::phash_dct64(&buf), format!("{:016x}", d.phash));
    // Not a thumbnail's: a 2× box-downscaled copy (a plausible shortcut) gives other PDQ bits.
    let (hw, hh) = ((w / 2) as usize, (h / 2) as usize);
    let mut half = Vec::with_capacity(hw * hh * 3);
    for y in 0..hh {
        for x in 0..hw {
            for c in 0..3 {
                let at = |dx: usize, dy: usize| u32::from(buf.rgb[((2 * y + dy) * w as usize + 2 * x + dx) * 3 + c]);
                half.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
            }
        }
    }
    let thumb = lightcraft_vera::digest_rgb8(hw, hh, &half).expect("thumb");
    assert_ne!(thumb.pdq, d.pdq);
}

// ---------------------------------------------------------------------------------------------
// CPNP-1 details the corpus does not pin
// ---------------------------------------------------------------------------------------------

/// The corpus's embedded profile is littleCMS's built-in sRGB: through it, every one of the
/// 2^24 8-bit colours maps to itself, as the profile measured under lcms 2.19.
#[test]
fn srgb_icc_is_identity_on_the_whole_cube() {
    let png = corpus_file("png-srgb-icc.png");
    let mut rgb = Vec::with_capacity(4096 * 4096 * 3);
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
    }
    // Splice the corpus's iCCP chunk into a 4096×4096 cube PNG.
    let cube = encode_png(4096, 4096, &rgb);
    let iccp = png_chunk(&png, b"iCCP").expect("iCCP");
    let mut tagged = cube[..33].to_vec();
    tagged.extend_from_slice(&iccp);
    tagged.extend_from_slice(&cube[33..]);
    let buf = cpnp::canonicalize(&tagged).expect("tagged cube");
    assert!(buf.icc_was_present);
    assert!(buf.rgb == rgb, "the sRGB → sRGB transform moved a colour");
}

fn png_chunk(png: &[u8], typ: &[u8; 4]) -> Option<Vec<u8>> {
    let mut p = 8;
    while p + 12 <= png.len() {
        let len = u32::from_be_bytes(png[p..p + 4].try_into().ok()?) as usize;
        if &png[p + 4..p + 8] == typ {
            return Some(png[p..p + 12 + len].to_vec());
        }
        p += 12 + len;
    }
    None
}

fn chunk(typ: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut c = (data.len() as u32).to_be_bytes().to_vec();
    c.extend_from_slice(typ);
    c.extend_from_slice(data);
    let mut h = crc32fast::Hasher::new();
    h.update(typ);
    h.update(data);
    c.extend_from_slice(&h.finalize().to_be_bytes());
    c
}

#[test]
fn unusable_icc_fails_closed_and_absent_icc_is_srgb() {
    let base = encode_png(4, 3, &[128u8; 36]);
    // A present profile that is not a profile: E8, no Canonical Buffer.
    let mut bad = base[..33].to_vec();
    let mut data = b"junk\0\0".to_vec();
    data.extend_from_slice(&miniz_compress(b"this is not an ICC profile"));
    bad.extend_from_slice(&chunk(b"iCCP", &data));
    bad.extend_from_slice(&base[33..]);
    assert!(matches!(cpnp::canonicalize(&bad), Err(cpnp::CpnpError::UnusableIcc(_))));
    // An iCCP stream that does not inflate is equally unusable.
    let mut broken = base[..33].to_vec();
    broken.extend_from_slice(&chunk(b"iCCP", b"junk\0\0\x78\x9c\xff\xff"));
    broken.extend_from_slice(&base[33..]);
    assert!(matches!(cpnp::canonicalize(&broken), Err(cpnp::CpnpError::UnusableIcc(_))));
    let ok = cpnp::canonicalize(&base).expect("untagged");
    assert!(!ok.icc_was_present);
    assert_eq!(ok.rgb, vec![128u8; 36]);
}

fn miniz_compress(b: &[u8]) -> Vec<u8> {
    // A stored (uncompressed) zlib stream, by hand: header, one final stored block, Adler-32.
    let mut out = vec![0x78, 0x01, 0x01];
    out.extend_from_slice(&(b.len() as u16).to_le_bytes());
    out.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
    out.extend_from_slice(b);
    let (mut a, mut s) = (1u32, 0u32);
    for &x in b {
        a = (a + u32::from(x)) % 65521;
        s = (s + a) % 65521;
    }
    out.extend_from_slice(&((s << 16) | a).to_be_bytes());
    out
}

/// A minimal ICC v2 gray display profile: D50 `wtpt` and a gamma-2.2 `kTRC`.
fn gray_profile() -> Vec<u8> {
    let s15 = |v: f64| ((v * 65536.0).round() as i32).to_be_bytes();
    let mut wtpt = b"XYZ \0\0\0\0".to_vec();
    for v in [0.9642, 1.0, 0.8249] {
        wtpt.extend_from_slice(&s15(v));
    }
    let ktrc = [b"curv\0\0\0\0".as_slice(), &1u32.to_be_bytes(), &563u16.to_be_bytes(), &[0, 0]].concat();
    let tags: [(&[u8; 4], &[u8]); 2] = [(b"wtpt", &wtpt), (b"kTRC", &ktrc)];
    let mut table = (tags.len() as u32).to_be_bytes().to_vec();
    let mut body = Vec::new();
    let mut off = 128 + 4 + 12 * tags.len();
    for (sig, data) in tags {
        table.extend_from_slice(sig.as_slice());
        table.extend_from_slice(&(off as u32).to_be_bytes());
        table.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
        off += data.len();
    }
    let size = 128 + table.len() + body.len();
    let mut h = vec![0u8; 128];
    h[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    h[8..12].copy_from_slice(&[2, 0x10, 0, 0]);
    h[12..16].copy_from_slice(b"mntr");
    h[16..20].copy_from_slice(b"GRAY");
    h[20..24].copy_from_slice(b"XYZ ");
    h[36..40].copy_from_slice(b"acsp");
    for (i, v) in [0.9642, 1.0, 0.8249].into_iter().enumerate() {
        h[68 + 4 * i..72 + 4 * i].copy_from_slice(&s15(v));
    }
    [h, table, body].concat()
}

/// A PNG of `color_type` (0 gray, 4 gray+alpha) with `data` rows and an `iCCP` carrying `profile`.
fn png_with_profile(w: u32, h: u32, color: png::ColorType, data: &[u8], profile: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().expect("header");
        wr.write_image_data(data).expect("data");
    }
    let mut iccp = b"gray\0\0".to_vec();
    iccp.extend_from_slice(&miniz_compress(profile));
    [&out[..33], &chunk(b"iCCP", &iccp), &out[33..]].concat()
}

/// Gray through a gray profile converts; gray + alpha is promoted to RGB before colour management
/// (as the reference does), where a gray profile no longer applies: E8, no buffer.
#[test]
fn gray_profiles_follow_the_reference() {
    let profile = gray_profile();
    let gray = cpnp::canonicalize(&png_with_profile(4, 1, png::ColorType::Grayscale, &[0, 64, 128, 255], &profile))
        .unwrap_or_else(|e| panic!("gray + gray profile: {e}"));
    assert!(gray.icc_was_present);
    // Gamma 2.2 → the sRGB curve, by lcms's 33-node resampling (measured equal to lcms 2.19).
    assert_eq!(gray.rgb, [0, 0, 0, 62, 62, 62, 129, 129, 129, 255, 255, 255]);
    let la = png_with_profile(2, 1, png::ColorType::GrayscaleAlpha, &[64, 255, 128, 255], &profile);
    assert!(matches!(cpnp::canonicalize(&la), Err(cpnp::CpnpError::UnusableIcc(_))));
}

#[test]
fn raw_and_unknown_inputs_are_refused_not_decoded() {
    assert!(matches!(cpnp::canonicalize(b""), Err(cpnp::CpnpError::Undecodable(_))));
    assert!(matches!(cpnp::canonicalize(b"hello, world"), Err(cpnp::CpnpError::Undecodable(_))));
    assert_eq!(cpnp::is_cpnp_format(b"hello"), None);
    assert_eq!(cpnp::is_cpnp_format(&corpus_file("jpeg-no-exif.jpg")), Some(Format::Jpeg));
    assert_eq!(cpnp::is_cpnp_format(&corpus_file("tiff-lzw.tiff")), Some(Format::Tiff));
    assert_eq!(Format::Webp.as_str(), "webp");
}

// ---------------------------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------------------------

fn app(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    v.extend_from_slice(payload);
    v
}

/// EXIF with IFD0 holding only tag 274 = `orientation` (big-endian).
fn exif_payload(orientation: u16) -> Vec<u8> {
    let mut v = b"Exif\0\0MM\0\x2a\0\0\0\x08\0\x01".to_vec();
    v.extend_from_slice(&[0x01, 0x12, 0x00, 0x03, 0, 0, 0, 1]);
    v.extend_from_slice(&orientation.to_be_bytes());
    v.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    v
}

#[test]
fn jpeg_segments_round_trip_byte_exact() {
    let base_buf = cpnp::canonicalize(&corpus_file("png-rgb.png")).expect("png");
    let base = encode_jpeg(&base_buf, 90);
    let profile = {
        let p = png_chunk(&corpus_file("png-srgb-icc.png"), b"iCCP").expect("iCCP");
        let data = &p[8..p.len() - 4];
        let nul = data.iter().position(|&b| b == 0).expect("name");
        miniz_oxide_inflate(&data[nul + 2..])
    };
    let exif = exif_payload(6);
    let mut xmp = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    xmp.extend_from_slice(br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/></x:xmpmeta>"#);
    let (a, b) = profile.split_at(profile.len() / 2);
    let icc1 = [b"ICC_PROFILE\0".as_slice(), &[1, 2], a].concat();
    let icc2 = [b"ICC_PROFILE\0".as_slice(), &[2, 2], b].concat();
    let iptc = [b"Photoshop 3.0\08BIM\x04\x04\0\0\0\0\0\x0b".as_slice(), b"\x1c\x02\x78\x00\x06Hello!\0"].concat();
    let mut jpeg = base[..2].to_vec();
    for (m, p) in [(0xE1, &exif), (0xE1, &xmp), (0xE2, &icc1), (0xE2, &icc2), (0xED, &iptc)] {
        jpeg.extend_from_slice(&app(m, p));
    }
    jpeg.extend_from_slice(&base[2..]);

    let (avail, segs) = segments::segments(&jpeg, Format::Jpeg);
    assert_eq!(
        (avail.exif, avail.xmp, avail.iptc, avail.icc, avail.c2pa),
        (State::Present, State::Present, State::Present, State::Present, State::Absent)
    );
    let ours: Vec<(&str, &[u8])> = segs.iter().filter(|s| s.family != "other").map(|s| (s.family, s.bytes.as_slice())).collect();
    assert_eq!(
        ours,
        vec![("exif", exif.as_slice()), ("xmp", xmp.as_slice()), ("icc", icc1.as_slice()), ("icc", icc2.as_slice()), ("iptc-iim", iptc.as_slice())]
    );
    // The encoder's JFIF APP0 is preserved as `other`.
    assert!(segs.iter().any(|s| s.family == "other" && s.bytes.starts_with(b"JFIF")));
    let entries = segments::raw_entries(&segs);
    for (e, s) in entries.iter().zip(&segs) {
        assert_eq!(b64::decode(e["bytes"].as_str().expect("bytes")).as_deref(), Some(s.bytes.as_slice()));
        assert_eq!(e["sha256"].as_str(), Some(sha256(&s.bytes).as_str()));
        assert_eq!(e["segment"].as_str(), Some(s.family));
    }

    // CPNP-1 consumed both: orientation 6 transposes, the two-chunk sRGB profile is the identity.
    let tagged = cpnp::canonicalize(&jpeg).expect("tagged");
    let plain = cpnp::canonicalize(&base).expect("plain");
    assert!(tagged.icc_was_present);
    assert_eq!((tagged.width, tagged.height), (plain.height, plain.width));
    let (w, h) = (plain.width as usize, plain.height as usize);
    for y in 0..w {
        for x in 0..h {
            let (sx, sy) = (y, h - 1 - x);
            assert_eq!(tagged.rgb[(y * h + x) * 3..][..3], plain.rgb[(sy * w + sx) * 3..][..3]);
        }
    }
}

fn miniz_oxide_inflate(z: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec_zlib(z).expect("inflate")
}

#[test]
fn png_segments_are_whole_chunks_and_absent_is_absent() {
    let base = encode_png(3, 2, &[10u8; 18]);
    let exif = chunk(b"eXIf", &exif_payload(3)[6..]);
    let iccp = png_chunk(&corpus_file("png-srgb-icc.png"), b"iCCP").expect("iCCP");
    let mut png = base[..33].to_vec();
    png.extend_from_slice(&iccp);
    png.extend_from_slice(&exif);
    png.extend_from_slice(&base[33..]);
    let (avail, segs) = segments::segments(&png, Format::Png);
    assert_eq!(
        (avail.exif, avail.xmp, avail.iptc, avail.icc, avail.c2pa),
        (State::Present, State::Absent, State::Absent, State::Present, State::Absent)
    );
    assert_eq!(segs.iter().map(|s| (s.family, s.bytes.clone())).collect::<Vec<_>>(), vec![("icc", iccp.clone()), ("exif", exif.clone())]);
    // Orientation 3 through eXIf.
    let buf = cpnp::canonicalize(&png).expect("png");
    assert_eq!((buf.width, buf.height), (3, 2));

    let (avail, segs) = segments::segments(&corpus_file("png-rgb.png"), Format::Png);
    assert!(segs.is_empty());
    for s in [avail.exif, avail.xmp, avail.iptc, avail.icc, avail.c2pa] {
        assert_eq!(s, State::Absent);
    }
    let (avail, _) = segments::segments(&corpus_file("tiff-lzw.tiff"), Format::Tiff);
    assert_eq!(avail.exif, State::Absent);
}

// ---------------------------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------------------------

#[test]
fn built_records_validate_under_both_schemas() {
    for name in ["jpeg-orientation-6.jpg", "png-srgb-icc.png", "png-rgba-alpha.png", "gif-frame0-subrectangle.gif"] {
        for schema in [Schema::V1_0_1, Schema::V1_1Draft] {
            let rec = l3_record(&corpus_file(name), schema).unwrap_or_else(|e| panic!("{name} {schema:?}: {e}"));
            let structural = validate::structural(&rec.0, schema);
            assert!(structural.is_empty(), "{name} {schema:?}: {structural:?}");
            let profile = validate::profile(&rec.0);
            assert!(validate::is_conformant(&profile), "{name} {schema:?}: {profile:?}");
            let codes: Vec<&str> = profile.iter().map(|i| i.code.as_str()).collect();
            assert!(codes.iter().all(|c| ["VER308", "VER404"].contains(c)), "{codes:?}");
            assert_eq!(rec.schema(), Some(schema));
            let id = rec.identity().expect("identity");
            assert_eq!(id.width, rec.0["image"]["width"].as_u64());
        }
    }
}

#[test]
fn builder_refuses_invalid_documents() {
    let bytes = corpus_file("png-rgb.png");
    let (buf, id) = hashing::identity(&bytes).expect("identity");
    // No canonical_visual embedding: VER201 / VER705.
    let r = record::Builder::new(Schema::V1_0_1, "t", "0").identity(&id, buf.format, false, false, false).space(space(4, "fp32")).build();
    assert!(r.is_err());
    // A claimed L1 without availability: VER303.
    let r = record::Builder::new(Schema::V1_0_1, "t", "0")
        .identity(&id, buf.format, false, false, false)
        .space(space(4, "fp32"))
        .embedding("canonical_visual", "org.lightcraft.space.test-visual/1.0", 4, "fp32", &[1.0, 0.0, 0.0, 0.0], "2026-10-11T10:00:00Z", "t")
        .level(record::Level::L1)
        .build();
    assert!(r.as_ref().is_err_and(|e| e.contains("VER303")), "{r:?}");
    // A wrong dim: VER604.
    let r = record::Builder::new(Schema::V1_0_1, "t", "0")
        .identity(&id, buf.format, false, false, false)
        .space(space(4, "fp32"))
        .embedding("canonical_visual", "org.lightcraft.space.test-visual/1.0", 4, "fp32", &[1.0, 0.0], "2026-10-11T10:00:00Z", "t")
        .build();
    assert!(r.as_ref().is_err_and(|e| e.contains("VER604")), "{r:?}");
    // Defaults: an E16 record_id and a UTC created_at.
    let r = record::Builder::new(Schema::V1_0_1, "t", "0")
        .identity(&id, buf.format, false, false, false)
        .space(space(4, "fp32"))
        .embedding("canonical_visual", "org.lightcraft.space.test-visual/1.0", 4, "fp32", &[3.0, 4.0, 0.0, 0.0], "2026-10-11T10:00:00Z", "t")
        .build()
        .expect("minimal L0");
    assert_eq!(r.0["record_id"].as_str(), Some(record::default_record_id("t", &id.content_hash, Schema::V1_0_1).as_str()));
    assert!(r.0["created_at"].as_str().is_some_and(|t| t.ends_with('Z')));
    // l2-normalised before packing.
    assert_eq!(r.embeddings()[0].4.as_deref(), Some(&[0.6f32, 0.8, 0.0, 0.0][..]));
}

#[test]
fn record_parse_and_embedding_decode() {
    let rec = l3_record(&corpus_file("png-rgb.png"), Schema::V1_0_1).expect("record");
    let back = Record::parse(&rec.to_json_pretty()).expect("parse");
    assert_eq!(back, rec);
    let e = back.embeddings();
    assert_eq!(e.len(), 1);
    let (role, space_id, dim, dtype, v) = &e[0];
    assert_eq!(
        (role.as_str(), space_id.as_str(), *dim, dtype.as_deref()),
        ("canonical_visual", "org.lightcraft.space.test-visual/1.0", Some(8), Some("fp16"))
    );
    let want = record::l2_normalize(&(0..8).map(|i| i as f32 - 3.5).collect::<Vec<_>>());
    let got = v.as_ref().expect("vector");
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-3, "{got:?} vs {want:?}");
    }
    assert_eq!(back.spaces().len(), 1);
    assert!(Record::parse("[1,2]").is_err());
    assert!(Record::parse("{\"a\":").is_err());
}

#[test]
fn vector_packing_matches_the_reference_sets_layout() {
    // fp16 little-endian, component 0 first, no header: 1.0 → 0x3C00, −2.0 → 0xC000, 0.5 → 0x3800,
    // 65504 → 0x7BFF, 1/3 → 0x3555 (round to nearest even).
    let v = [1.0f32, -2.0, 0.5, 65504.0, 1.0 / 3.0];
    let packed = record::pack_vector(&v, "fp16");
    assert_eq!(packed, vec![0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0xFF, 0x7B, 0x55, 0x35]);
    assert_eq!(record::unpack_vector(&packed, "fp16")[..4], [1.0, -2.0, 0.5, 65504.0]);
    // vector_sha256 is SHA-256 over exactly those octets.
    assert_eq!(sha256(&packed), hashing::sha256_hex(&[0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0xFF, 0x7B, 0x55, 0x35]));
    assert_eq!(record::pack_vector(&[1.5], "fp32"), 1.5f32.to_le_bytes().to_vec());
    assert_eq!(record::pack_vector(&[1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], "binary"), vec![0b1010_0001, 0b1000_0000]);
    assert_eq!(record::pack_vector(&[1.0], "bf16"), vec![0x80, 0x3F]);
    assert_eq!(record::pack_vector(&[-3.6, 200.0], "int8"), vec![0xFC, 0x7F]);
}

#[test]
fn uuid_v5_matches_rfc_9562() {
    // RFC 9562 appendix A.4: UUIDv5 of "www.example.com" in the DNS namespace.
    let dns = [0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8];
    assert_eq!(record::uuid_v5(&dns, b"www.example.com"), "2ed6657d-e927-568b-95e1-2665a8aea6a2");
}

// ---------------------------------------------------------------------------------------------
// Conformance fixtures
// ---------------------------------------------------------------------------------------------

fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(standard().join("conformance/manifest.json")).expect("manifest")).expect("json")
}

fn schema_of(entry: &Value) -> Schema {
    match entry.get("schema").and_then(Value::as_str) {
        Some("1.1-draft") => Schema::V1_1Draft,
        _ => Schema::V1_0_1,
    }
}

#[test]
fn valid_fixtures_are_structurally_clean() {
    let m = manifest();
    let mut n = 0;
    for e in m["valid"].as_array().expect("valid") {
        let file = e["file"].as_str().expect("file");
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(standard().join("conformance").join(file)).expect("read")).expect("json");
        let issues = validate::structural(&doc, schema_of(e));
        assert!(issues.is_empty(), "{file}: {issues:?}");
        let errors: Vec<_> = validate::profile(&doc).into_iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errors.is_empty(), "{file}: {errors:?}");
        n += 1;
    }
    assert_eq!(n, 14);
}

#[test]
fn schema_stage_invalid_fixtures_fail_structurally() {
    let m = manifest();
    let mut checked = 0;
    for e in m["invalid"].as_array().expect("invalid") {
        let file = e["file"].as_str().expect("file");
        let codes: Vec<&str> = e["expect_codes"].as_array().expect("codes").iter().filter_map(Value::as_str).collect();
        if !codes.iter().any(|c| validate::STRUCTURAL_CODES.contains(c) || *c == "VER101") {
            continue;
        }
        let path = standard().join("conformance").join(file);
        let text = std::fs::read_to_string(&path).expect("read");
        let Ok(doc) = serde_json::from_str::<Value>(&text) else {
            assert!(codes.contains(&"VER101"), "{file} is not JSON but does not expect VER101");
            assert!(Record::parse(&text).is_err());
            checked += 1;
            continue;
        };
        let issues = validate::structural(&doc, schema_of(e));
        assert!(issues.iter().any(|i| i.severity == Severity::Error), "{file}: expected a structural error for {codes:?}");
        for c in codes.iter().filter(|c| validate::STRUCTURAL_CODES.contains(c)) {
            assert!(issues.iter().any(|i| i.code == *c), "{file}: missing {c}: {issues:?}");
        }
        checked += 1;
    }
    assert!(checked >= 15, "{checked}");
}

#[test]
fn profile_reports_every_claimed_expected_code() {
    let m = manifest();
    let mut checked = 0;
    for e in m["invalid"].as_array().expect("invalid") {
        let file = e["file"].as_str().expect("file");
        let Ok(text) = std::fs::read_to_string(standard().join("conformance").join(file)) else { continue };
        let Ok(doc) = serde_json::from_str::<Value>(&text) else { continue };
        let level = e.get("level").and_then(Value::as_str).and_then(|l| match l {
            "L0" => Some(record::Level::L0),
            "L1" => Some(record::Level::L1),
            "L2" => Some(record::Level::L2),
            "L3" => Some(record::Level::L3),
            _ => None,
        });
        let mut issues = validate::structural(&doc, schema_of(e));
        issues.extend(validate::profile_at(&doc, level));
        for c in e["expect_codes"].as_array().expect("codes").iter().filter_map(Value::as_str) {
            if validate::PROFILE_CODES.contains(&c) || validate::STRUCTURAL_CODES.contains(&c) {
                assert!(issues.iter().any(|i| i.code == c), "{file}: expected {c}, got {:?}", issues.iter().map(|i| &i.code).collect::<Vec<_>>());
                checked += 1;
            }
        }
    }
    assert!(checked >= 50, "{checked}");
}

// ---------------------------------------------------------------------------------------------
// Never crash
// ---------------------------------------------------------------------------------------------

#[test]
fn hostile_inputs_never_panic() {
    let mut seeds: Vec<Vec<u8>> = Vec::new();
    for name in [
        "jpeg-q75-420.jpg",
        "jpeg-orientation-6.jpg",
        "png-rgba-alpha.png",
        "png-srgb-icc.png",
        "gif-frame0-subrectangle.gif",
        "tiff-lzw.tiff",
        "webp-lossless.webp",
    ] {
        seeds.push(corpus_file(name));
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut cases = 0;
    for i in 0..200 {
        let seed = &seeds[i % seeds.len()];
        let mut b = seed.clone();
        match i % 4 {
            0 => b.truncate((next() as usize) % seed.len()),
            1 => {
                for _ in 0..1 + next() % 8 {
                    let at = (next() as usize) % b.len();
                    b[at] = next() as u8;
                }
            }
            2 => {
                let at = 2 + (next() as usize) % (b.len() - 2);
                b.truncate(at);
                b.extend((0..next() % 64).map(|_| next() as u8));
            }
            _ => b = (0..next() % 512).map(|_| next() as u8).collect(),
        }
        let _ = cpnp::canonicalize(&b);
        let _ = cpnp::is_cpnp_format(&b);
        for f in [Format::Jpeg, Format::Png, Format::Gif, Format::Tiff, Format::Webp, Format::Bmp] {
            let _ = segments::segments(&b, f);
        }
        let _ = Record::parse(&String::from_utf8_lossy(&b));
        cases += 1;
    }
    assert_eq!(cases, 200);
    for text in
        ["", "{", "null", "{\"embeddings\": 3, \"spaces\": [1, null]}", "{\"identity\": {\"perceptual\": [{\"alg\": \"pdq/1.0\", \"value\": 5}]}}"]
    {
        if let Ok(r) = Record::parse(text) {
            let _ = validate::all(&r.0, Schema::V1_0_1);
            let _ = validate::all(&r.0, Schema::V1_1Draft);
            let _ = r.embeddings();
            let _ = r.identity();
        }
    }
}

/// Writes every decoder-corpus Canonical Buffer as an 8-bit RGB PNG into `$VER_EXPORT_DIR`, the
/// input `crates/vera/tests/gen_perceptual_vectors.py` feeds the reference perceptual hashers.
#[test]
#[ignore = "generator input; run with VER_EXPORT_DIR set"]
fn export_canonical_pngs() {
    let Ok(dir) = std::env::var("VER_EXPORT_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("export dir");
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(standard().join("decoder-corpus/1/manifest.json")).expect("manifest")).expect("json");
    for f in m["files"].as_array().expect("files") {
        let path = f["path"].as_str().expect("path");
        let buf = cpnp::canonicalize(&std::fs::read(standard().join("decoder-corpus/1").join(path)).expect("read")).expect("canonical");
        std::fs::write(dir.join(format!("{}.png", path.replace('/', "__"))), encode_png(buf.width, buf.height, &buf.rgb)).expect("write");
    }
}

#[derive(serde::Deserialize)]
struct PerceptualVector {
    path: String,
    pdq: String,
    pdq_quality: u8,
    phash: String,
}

/// The reference perceptual digests (errata E22: Meta's PDQ, `imagehash.phash`) of every corpus
/// Canonical Buffer and reference-set image, as the Python references computed them.
#[test]
fn perceptual_digests_match_the_reference_vectors() {
    let file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vera/tests/perceptual-vectors.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(file).expect("vectors")).expect("json");
    let vectors: Vec<PerceptualVector> = serde_json::from_value(doc["vectors"].clone()).expect("vectors");
    let assets: Vec<PerceptualVector> = vectors.into_iter().filter(|v| !v.path.starts_with("synthetic:")).collect();
    assert_eq!(assets.len(), 31);
    for v in assets {
        let bytes = std::fs::read(standard().join(&v.path)).expect("asset");
        let buf = cpnp::canonicalize(&bytes).expect("canonical");
        let p = hashing::perceptual(&buf);
        assert_eq!(p.phash, v.phash, "{} phash", v.path);
        assert_eq!(p.pdq, v.pdq, "{} pdq", v.path);
        assert_eq!(p.pdq_quality, v.pdq_quality, "{} pdq quality", v.path);
        assert_eq!(hashing::pdq(&buf), v.pdq);
        assert_eq!(hashing::phash_dct64(&buf), v.phash);
    }
}

/// Per-step milliseconds of the Canonical-Buffer lane on a 24 MP (6000×4000) photo-like JPEG.
/// `cargo test -p lightcraft-ver --test integration timing_24mp -- --ignored --nocapture`.
#[test]
#[ignore = "timing; run with --ignored --nocapture"]
fn timing_24mp() {
    use std::time::Instant;
    let (w, h) = (6000u32, 4000u32);
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            rgb.extend_from_slice(&[((x * 255) / w) as u8, ((y * 255) / h) as u8, ((((x / 40) ^ (y / 40)) * 37) as u8).wrapping_add((s & 15) as u8)]);
        }
    }
    let mut jpeg = Vec::new();
    jpeg_encoder::Encoder::new(&mut jpeg, 90).encode(&rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb).expect("encode");
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let t = Instant::now();
    let buf = cpnp::canonicalize(&jpeg).expect("canonical");
    let decode = ms(t);
    let t = Instant::now();
    let planes = hashing::luma(&buf);
    let luma = ms(t);
    let t = Instant::now();
    let (_, quality) = planes.as_ref().and_then(lightcraft_vera::digest::pdq_planes).expect("pdq");
    let pdq = ms(t);
    let t = Instant::now();
    let _ = planes.as_ref().and_then(lightcraft_vera::digest::phash_planes).expect("phash");
    let phash = ms(t);
    let t = Instant::now();
    let _ = hashing::pixel_hash(&buf);
    let pixel = ms(t);
    let t = Instant::now();
    let _ = hashing::content_hash(&jpeg);
    let content = ms(t);
    let total = ms(t0);
    println!(
        "24 MP: CPNP-1 decode {decode:.0} ms, luma planes {luma:.0} ms, PDQ + quality ({quality}) {pdq:.0} ms, pHash {phash:.0} ms, pixel_hash {pixel:.0} ms, content_hash {content:.0} ms, total {total:.0} ms"
    );
    let t = Instant::now();
    let _ = hashing::identity_with_quality(&jpeg).expect("identity");
    println!("hashing::identity_with_quality end to end: {:.0} ms", ms(t));
}
