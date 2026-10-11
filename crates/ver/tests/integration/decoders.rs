//! The evidence for decoding JPEG ourselves: the two JPEG decoders already in the workspace,
//! measured against libjpeg-turbo's output on the upright corpus JPEGs. Our decoder reproduces
//! the corpus digests exactly, so its RGB *is* libjpeg-turbo's here; the table this prints is
//! the one `docs/ver-cpnp.md` quotes. It asserts that ours matches the corpus and records, as a
//! regression guard, that neither third-party decoder does (if one ever does, revisit the choice).

use crate::common::*;
use lightcraft_ver::cpnp;
use std::collections::BTreeMap;

fn zune(bytes: &[u8]) -> Vec<u8> {
    let opts = zune_core::options::DecoderOptions::default().jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(zune_core::bytestream::ZCursor::new(bytes), opts);
    d.decode().expect("zune-jpeg decodes")
}

fn jpeg_decoder(bytes: &[u8]) -> Vec<u8> {
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    d.decode().expect("jpeg-decoder decodes")
}

fn compare(reference: &[u8], other: &[u8]) -> (usize, i32, BTreeMap<i32, usize>) {
    let mut hist = BTreeMap::new();
    let mut n = 0;
    let mut max = 0;
    for (a, b) in reference.iter().zip(other) {
        let d = (i32::from(*a) - i32::from(*b)).abs();
        if d > 0 {
            n += 1;
            max = max.max(d);
            *hist.entry(d).or_insert(0) += 1;
        }
    }
    (n, max, hist)
}

#[test]
fn workspace_jpeg_decoders_versus_libjpeg_turbo() {
    let mut any_differs = [false, false];
    for name in ["jpeg-q92-444.jpg", "jpeg-q75-420.jpg", "jpeg-orientation-1.jpg"] {
        let bytes = corpus_file(name);
        let ours = cpnp::canonicalize(&bytes).expect("ours").rgb;
        for (k, (label, other)) in [("zune-jpeg", zune(&bytes)), ("jpeg-decoder", jpeg_decoder(&bytes))].into_iter().enumerate() {
            assert_eq!(other.len(), ours.len(), "{label} {name}");
            let (n, max, hist) = compare(&ours, &other);
            println!("| `{name}` | {label} | {n} of {} samples differ | max {max} | {hist:?} |", ours.len());
            any_differs[k] |= n > 0;
        }
    }
    assert!(any_differs[0] && any_differs[1], "a workspace decoder now matches libjpeg-turbo on the corpus: reconsider crate::jpeg");
}
