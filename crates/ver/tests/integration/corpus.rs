//! The published decoder corpus and the reference-set pixel vectors, from the vendored copies.

use crate::common::standard;
use lightcraft_ver::corpus;

#[test]
fn decoder_corpus() {
    let report = corpus::run_decoder_corpus(&standard().join("decoder-corpus/1")).expect("corpus runs");
    println!("{}", corpus::render_evaluation(&report));
    assert_eq!(report.total, 21);
    assert!(report.entries.iter().all(|e| !e.note.contains("content_sha256")), "vendored corpus bytes drifted");
    assert_eq!(report.reproduced, corpus::PINNED_DECODER_CORPUS);
}

#[test]
fn reference_set_pixels() {
    let report = corpus::run_reference_set_pixels(&standard().join("reference-sets")).expect("refsets run");
    println!("{}", corpus::render_evaluation(&report));
    assert_eq!(report.total, 10);
    assert_eq!(report.reproduced, corpus::PINNED_REFERENCE_SET_PIXELS);
}
