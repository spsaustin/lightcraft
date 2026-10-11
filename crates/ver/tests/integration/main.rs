//! lightcraft-ver integration tests, one binary (each test binary links the whole decode stack).
//!
//! - [`corpus`]: the published decoder corpus and the reference-set pixel vectors.
//! - [`ver`]: hashes, perceptual digests, segments, records, conformance fixtures, never-crash.
//! - [`decoders`]: why JPEG is decoded by `lightcraft_ver` itself.
//! - [`oracle`]: agreement with the upstream Python validator (needs `VER_ORACLE_PY`/`VER_ORACLE_SRC`).

mod common;
mod corpus;
mod decoders;
mod oracle;
mod ver;
