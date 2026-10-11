//! The VER standard's core for LightCraft (layer L3).
//!
//! - [`cpnp`]: CPNP-1, the canonical pixel normalisation procedure (bytes in, one 8-bit sRGB
//!   Canonical Buffer out), executed as the published implementation profile `verd-pillow/1`
//!   pins it.
//! - [`hashing`]: `content_hash`, `pixel_hash` and the two registered perceptual digests.
//! - [`segments`]: byte-exact metadata segments (errata E11) and their availability (E21).
//! - [`record`]: VER Records (1.0.1 and the 1.1 draft): a builder that refuses to emit an
//!   invalid document, and a parser.
//! - [`validate`]: structural validation against the vendored schemas plus the cross-object
//!   profile rules, coded as the upstream conformance validator codes them.
//! - [`corpus`]: the published decoder corpus and the reference-set pixel vectors, and the
//!   evaluation document they produce.
//!
//! The vendored standard files live in `crates/ver/standard/` (see `README.md` there).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod b64;
mod cms;
pub mod corpus;
pub mod cpnp;
pub mod hashing;
mod jpeg;
pub mod record;
pub mod segments;
mod tiffifd;
pub mod time;
pub mod validate;
