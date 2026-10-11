# lightcraft-ver

The VER standard's core for LightCraft (layer L3): CPNP-1 canonicalisation, the identity hashes, VER
Records (artifact release 1.0.1 and the 1.1 draft), structural and profile validation, and the
published decoder corpus. Pure Rust, no network, no panics on hostile input.

The evaluation against the published decoder corpus, and every decision where the profile leaves
room, is in [`docs/ver-cpnp.md`](../../docs/ver-cpnp.md): **21 of 21** entries of
`ver-decoder-corpus/1` reproduce (profile `verd-pillow/1`), and all ten reference-set images.

## API

```rust
use lightcraft_ver::{cpnp, hashing, segments, record, validate, corpus};

// CPNP-1: bytes in, one 8-bit sRGB Canonical Buffer out (or why not).
let buf = cpnp::canonicalize(&bytes)?;            // CanonicalBuffer { width, height, rgb, format, animated, alpha_present, hdr_present, icc_was_present }
cpnp::is_cpnp_format(&bytes);                      // Option<Format>; None for camera raws and unknowns
// CpnpError::{Undecodable, UnusableIcc (errata E8: fail closed), Unsupported (raw, AVIF, HEIC)}

// Identity: content_hash, pixel_hash (VER1:w:h:sRGB8:…), pdq/1.0 (64 hex), phash-dct/64 (16 hex);
// the perceptual digests are bit-exact with Meta's PDQ and imagehash.phash (docs/ver-cpnp.md).
let (buf, id) = hashing::identity(&bytes)?;        // Identity { content_hash, pixel_hash, width, height, pdq, phash }
hashing::identity_with_quality(&bytes)?;           // … plus PDQ's quality figure, one decode, one luma pass
let planes = hashing::luma(&buf);                  // Option<Planes>: PDQ's f32 luma + PIL's L, built once
hashing::perceptual_from(planes.as_ref());         // Perceptual { pdq, pdq_quality, phash }; pdq_from / phash_from too

// Metadata segments, byte-exact per errata E11, and availability per E21.
let (availability, raw) = segments::segments(&bytes, buf.format);
segments::raw_entries(&raw);                       // schema-valid metadata.raw[] entries

// Records: the builder validates (structural + profile) and refuses to emit errors.
let rec = record::Builder::new(record::Schema::V1_0_1, "lightcraft", "0.5.0")
    .identity(&id, buf.format, buf.animated, buf.alpha_present, buf.hdr_present)
    .availability(&availability).raw_segments(&raw)
    .space(descriptor_json)
    .embedding("canonical_visual", space_id, dim, "fp16", &vector, produced_at, producer)
    .level(record::Level::L1)
    .build()?;                                      // Record(serde_json::Value)
record::Record::parse(&text)?.embeddings();        // (role, space_id, dim, dtype, Option<Vec<f32>>)
record::pack_vector(&v, "fp16"); record::unpack_vector(&bytes, "fp16");

// Validation, coded as the upstream validator codes it.
validate::structural(&doc, record::Schema::V1_1Draft);   // VER102 / VER201 / VER202
validate::profile(&doc);                                  // VER3xx–VER13xx, see validate::PROFILE_CODES
validate::all(&doc, schema); validate::is_conformant(&issues);

// The published test vectors.
corpus::run_decoder_corpus(dir)?; corpus::run_reference_set_pixels(dir)?; corpus::render_evaluation(&report);
```

Additions beyond the contract the engine codes against: `Format::from_token`,
`State::{StrippedSuspected, Redacted, UpstreamWithheld}` + `as_str`/`from_token`,
`Availability::to_json`, `hashing::{sha256_hex, luma, Planes, Perceptual, perceptual, perceptual_from,
pdq_from, phash_from, pdq_quality, identity_with_quality}`, `record::{IdentityView, EmbeddingEntry,
l2_normalize, uuid_v5, default_record_id, lightcraft_namespace}`, `Schema::from_ver_version`,
`Level::as_str`, `validate::{profile_at, schema_source, pointer, STRUCTURAL_CODES, PROFILE_CODES,
LOCAL_CODES}`, `corpus::{toolchain, PINNED_DECODER_CORPUS, PINNED_REFERENCE_SET_PIXELS,
DECODER_CORPUS_DIR, REFERENCE_SETS_DIR}`, `time::{now_rfc3339, format_utc}`, `b64::{encode, decode}`.

## How CPNP-1 is executed

- **JPEG** is decoded by this crate's own decoder (`src/jpeg.rs`): baseline, extended and progressive
  Huffman, 8-bit, 1/3/4 components, restart intervals, libjpeg's damaged-data behaviour; the islow
  integer IDCT, libjpeg-turbo's fancy `h2v1`/`h2v2`/`h1v2` upsampling and `jdcolor.c`'s fixed-point
  YCbCr tables. The two JPEG decoders already in the workspace (`zune-jpeg`, `jpeg-decoder`) both
  miss every JPEG entry of the corpus (see the doc for the measurements).
- **PNG** `png`, **WebP** `image-webp`, **TIFF** `tiff`, **GIF** `gif` (frame 0 coalesced on a
  transparent canvas, errata E7), **BMP** `image`. AVIF, HEIC, camera raws: `Unsupported`.
- **ICC → sRGB** (`src/cms.rs`): matrix/TRC RGB and gray TRC profiles are evaluated the way
  littleCMS evaluates its 8-bit transforms (bit-exact with lcms 2.19 on every case measured,
  including the full 2²⁴ cube through Display-P3-like and Adobe-RGB-like profiles); other profiles
  go through `moxcms` in `f32` with one rounding.

## Vendored standard files

`standard/` holds byte-identical copies of the VER standard's published files (Apache-2.0; source
and dates in [`standard/NOTICE`](standard/NOTICE)): `decoder-corpus/1/`, `reference-sets/`,
`schemas/` (both record schemas), `conformance/` (minus two oversized size-policy fixtures). The
tests read only these, so CI needs no network. A test pins the schemas' and the corpus manifest's
SHA-256.

**Re-vendoring.** From a vera checkout `$VERA`:

```sh
D=crates/ver/standard
cp -r $VERA/standards/decoder-corpus/1 $D/decoder-corpus/          # a new release is a new directory (2/)
cp -r $VERA/standards/reference-sets/{README.md,images,clip-vit-b32,siglip-base-p16-256,siglip2-base-p16-256} $D/reference-sets/
cp $VERA/standards/ver/1.0.1/ver-record.schema.json $D/schemas/ver-record-1.0.1.schema.json
cp $VERA/standards/ver/1.1-draft/ver-record.schema.json $D/schemas/ver-record-1.1-draft.schema.json
cp -r $VERA/conformance/{README.md,manifest.json,valid,invalid,keys,lineage-peers,registries} $D/conformance/
rm $D/conformance/invalid/ver10[45]-*.json
```

then update `standard/NOTICE`, the digests in `tests/integration/ver.rs`
(`vendored_schemas_are_byte_identical_to_upstream`), and run `cargo xtask ver corpus`,
`cargo xtask ver oracle` and `cargo xtask ver evaluation --write`.

## Tooling

- `cargo xtask ver corpus` / `ver refsets`: the evaluations (both in `cargo xtask ci`).
- `cargo xtask ver validate [--oracle auto|required|off] FILE…`: our validation, then the upstream
  Python validator, found in this order: `VER_ORACLE_PY` (an interpreter) with `VER_ORACLE_SRC`
  (`packages/ver-validator/src`); `VERA_CHECKOUT`, a vera checkout (its
  `packages/ver-validator/src`, run by its `.venv/bin/python`, else by `python3` on `PATH` if that
  imports `ver_validator`); a sibling checkout `../vera`, derived the same way.
- `cargo xtask ver oracle`: the oracle-agreement tests (`tests/integration/oracle.rs`).
- `cargo xtask ver evaluation --write`: regenerates the generated section of `docs/ver-cpnp.md`.
