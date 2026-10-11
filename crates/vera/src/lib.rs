//! VERA: the algorithms behind *find by words*, *find by meaning*, *same picture* and *similar picture*.
//!
//! Ported as a design from the VERA studio (the Renee Lynn Photography archive, `docs/VERA-SEARCH.md`
//! there; `docs/vera.md` here): the lane structure, the fusion, the measured thresholds and the
//! fixtures that chose them. Everything in this crate is a pure function over slices: no files, no
//! threads, no models, no knowledge of the catalog. The engine decides what to digest, where to keep
//! the results and when to run; this crate only computes.
//!
//! Modules, in the order a search uses them:
//! - [`index`]: the keyword lane, a small BM25 inverted index over a photo's words.
//! - [`digest`]: the identity lane, PDQ and pHash digests with Hamming distance and the
//!   same-picture decision.
//! - [`cutoff`]: where meaning ends and noise begins in a list of cosine scores.
//! - [`fuse`]: reciprocal rank fusion of ranked lists.
//! - [`thresholds`]: every number the proofs measured, each with its reason.
//!
//! Input-derived numbers are hostile everywhere here (image sizes, record counts, scores from a
//! file): nothing indexes without `get`, nothing divides without a guard, nothing allocates a size a
//! caller did not bound.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![deny(clippy::indexing_slicing)]

pub mod cutoff;
pub mod digest;
pub mod fuse;
pub mod index;
pub mod space;
pub mod thresholds;

pub use digest::{
    ALGORITHM_VERSION, Comparison, Digest, Luma, Planes, Sameness, compare, digest, digest_planes, digest_rgb8, distances, hamming64, hamming256,
    luma_from_rgb, sameness,
};
pub use fuse::rrf;
pub use index::{Hit, Index};
pub use space::Space;
