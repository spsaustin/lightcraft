# VER: LightCraft's Visual Embedding Records

LightCraft produces and reads **VER Records** (the Visual Embedding Record standard, `verstandard.org`): one JSON
document per picture that says what the picture is (its identity over the CPNP-1 Canonical Buffer), what metadata it
carried and how far to trust it, where it came from and what was done to it, and where its embedding lives. VER is the
product here, not an export format: every VERA feature that fingerprints or embeds a picture does it the way VER
defines, and every Record LightCraft hands out has been validated against its schema and the VERA profile first.

- **Core**: `lightcraft-ver` (`crates/ver`, layer 3): CPNP-1, `content_hash` / `pixel_hash` / PDQ / pHash, the
  byte-exact metadata segments, the Record builder, the schema and profile validators, the decoder corpus. Its
  vendored standard files are in `crates/ver/standard/`.
- **Engine**: `crates/engine/src/ver/` (producing, consuming, the export hook), the commands in
  `crates/engine/src/cmd/ver.rs`, and the digest lane in `crates/engine/src/vera/` (*The digest lane* in
  [vera.md](vera.md)).
- **MCP**: `ver_record`, `ver_import`, `ver_export`, `ver_status` ([mcp.md](mcp.md)).

The standard's documents this follows: the VER 1.0.1 schema and errata annex (E1–E29), the VER 1.1 draft (§1 the
migration, §2.3 provenance actions, §3 lineage, §8 reference sets, §9 compute, §10 metadata carriage, §11.6 space ids,
§12 `dim`/`dtype`), the registry snapshot 1 and the reference-set README.

## What a Record carries, level by level

A Record claims the highest conformance level its evidence supports (and no higher than asked: `ver.record {level}`).
The level is a claim a consumer relies on, so each one is earned, never assumed.

| Level | What LightCraft puts in | Why it can |
|---|---|---|
| **L0** | `identity`: `content_hash` (SHA-256 of the file), `pixel_hash` (SHA-256 of the Canonical Buffer, errata E10, with its width and height), `perceptual`: `pdq/1.0` and `phash-dct/64` over the Canonical Buffer (E22). `image`: format, the Canonical Buffer's size (E2), animation / alpha / HDR flags, `source_bytes_ref` `{uri: file://…, sha256 = content_hash, byte_length}` (E25). `spaces` + one `canonical_visual` embedding (below) | CPNP-1 ran on the bytes |
| **L1** | `metadata.availability` for all five families, and `metadata.raw`: every stored segment byte-exact, as the container stores it (E11's per-container table), in file order, one entry per segment, each with its per-family `index` in 1.1 (§10.1) | The segment extractor reads containers, not a parsed view; "present" always means "carried" (E21) |
| **L2** | `metadata.normalized`: the IPTC Photo Metadata view in XMP spelling (`dc:title`, `dc:description`, `Iptc4xmpCore:AltTextAccessibility`, `Iptc4xmpCore:ExtDescrAccessibility`, `dc:subject`, `dc:creator`, `dc:rights`, `photoshop:DateCreated`, `Iptc4xmpCore:Location`, `photoshop:City` / `State` / `Country`), one `metadata.trust` entry per assertion, `metadata.conflicts` where sources disagree | At least one descriptive field is known (else the Record stops at L1 and says so in `notes`) |
| **L3** | `provenance.acquisition` (`user_upload`, the catalog's import time, the file's URI), `provenance.chain` (registered actions only: `import`, `decode`, `normalize`, `embed`; exports add `color_adjust`, `crop`, `resize`, `denoise` or `transcode`, then `export`), `provenance.c2pa.status` | An honest C2PA status exists (below) |
| signature | `provenance.signature`: a detached JWS, EdDSA over the RFC 8785 form of the Record without the signature (E17), and a `sign` chain event (1.1) | `ver.config {signKeyPath}` names an Ed25519 key; otherwise the Record is unsigned and the validator says so (VER308, a warning) |

**Reconciliation and trust (L2).** The file's own metadata wins, in E14's order **XMP > IPTC-IIM > EXIF**; what only
the catalog knows (a title typed in LightCraft, a caption accepted from Describe) is a *platform* value: it fills a
field the file leaves empty, never wins against the file, and is listed in `conflicts` with `source: "platform"` when
it disagrees. **Every trust tier is `unsigned`.** The schema's tiers (`c2pa_verified`, `signed_unverified`,
`unsigned`) grade cryptographic evidence; LightCraft verifies no C2PA manifest, and E14 says a platform value "always
carries trust tier `unsigned`". Where each value came from inside LightCraft (typed by the user, accepted from a
Describe model, read from the file) is not a trust tier, so 1.1 Records carry it in the `extensions` block instead:
`extensions["<authority>.lightcraft"].field_origins` (`xmp`, `iptc-iim`, `exif`, `platform:user`,
`platform:describe:<model>`, `platform:catalog`), with the `producer_id`.

**C2PA (L3, errata E28).** No segment → `missing`. A manifest present → in 1.1 `unverifiable` with the reason
`<authority>/x-validation_not_implemented` (LightCraft does not validate manifests, and `unverifiable` is never
positive evidence); in 1.0, which has no such status, `provenance.c2pa` is left out and the Record stops at L2 rather
than state a status contrary to fact. Presence unknown → no status, L2.

**Record ids (E16).** UUIDv5 in the namespace `uuid5(DNS, <authority's domain>)` over `producer | content_hash |
ver_version`: re-issuing a Record of the same bytes is idempotent, two producers never collide, and the 1.0 and 1.1
twins are two documents. The producer is this library's `did:ver:local:<uuid>`, minted once and kept in `prefs.json`.

**Schemas.** `1.1-draft` is the default for `ver.record` (`ver.config {defaultSchema}` changes it); exports and
`ver.export` dual-publish (VER 1.1 §1.1): a 1.0 Record and a 1.1 Record of the same bytes, sharing `content_hash`. The
1.0 twin carries only what 1.0 has: the six core chain actions (develop and export events are omitted, never spelled
as `transcode`), no `lineage`, `compute` or `extensions`. VER 1.1 is a draft: its banner asks that 1.1 Records not be
published to third parties outside a ratification trial, which is why the 1.0 twin is always written beside an export.

## The embedding and its space

A Record must carry a `canonical_visual` embedding "computed from the Canonical Buffer and nothing else" (VER 1.0 §6).
LightCraft computes it from the Canonical Buffer itself: resized to 256 × 256 by a bilinear (triangle) filter whose
support widens with the downscale (Pillow's `BILINEAR`, the filter the SigLIP processor uses), no crop, `/255`, mean and
standard deviation 0.5, through the vector lane's SigLIP 2 model, l2-normalised and stored as fp16 little-endian (the
reference-set packing). It is **not** the vector lane's stored search vector: that one is made from a default-settings
render of the photo, which is not the Canonical Buffer.

The space is LightCraft's own: `<authority>.space.siglip2-base-p32-256/1.0`, kind `visual`, model `{family: siglip2,
architecture: ViT-B/32, checkpoint: google/siglip2-base-patch32-256, weights_sha256}`, dim 768, fp16, l2, cosine, the
preprocessing members the reference producer (`verd`) names for its SigLIP 2 processor (`resize`, `do_resize`,
`interpolation`, `antialias`, `crop`, `do_center_crop`, `do_rescale`, `rescale_factor`, `do_normalize`, `mean`, `std`,
`channel_order`), provenance `{publisher: <authority's domain>, published_at, license: Apache-2.0}`. 1.1 embeddings add
`compute` (§9: `candle`, its version, fp32, cpu or gpu). `weights_sha256` is SHA-256 of `model.safetensors`
(E29: comparable only within one producer), computed once per weights file and remembered beside it.

**No `reference` block.** No tolerance has been measured for this space, so its descriptor makes no conformance claim
(1.1 §8.2, E13), its vectors must not be advertised as interoperable, and the validator warns `VER404` on every Record:
that warning is the honest state, not a defect. The published space with a reference set is
`org.verstandard.space.siglip2-base-p16-256/1.0` (another checkpoint); see *Not done yet*.

**Contextual embeddings** are not emitted. The vector lane's text vectors live in the same SigLIP 2 geometry, which a
1.0 Record can only declare as `visual` (a contextual embedding there is VER702) and which a second `space_id` over the
same weights would misdeclare (1.1 §11.1). A 1.1 `joint_visual_text` space needs the text half of the preprocessing
(tokenizer and its hash, max length, truncation, casing) stated too; that is follow-up work.

Without the model there is no Record (`needs_model`): a Record without its canonical visual embedding is not a Record.

## The authority

Space ids and the record-id namespace are minted under a reverse-DNS **authority the publisher controls** (1.1 §11.6:
"a Producer MUST NOT mint identifiers under an authority it does not control"). The operator of this fork controls
`verstandard.org`, so the default is `org.verstandard`. Anyone else distributing LightCraft sets their own with
`ver.config {authority: "com.example.photos"}` (lowercase, at least two labels, no hyphen in the first: the 1.0.1
pattern). Changing it changes the space id, so vectors made under the old one are a different space.

## Raw files have no Record

A camera raw (and anything else without a CPNP-1 path: an unknown format, an image whose embedded ICC profile CPNP-1
cannot use, E8) has no Canonical Buffer, so it has no `pixel_hash` and **no Record of its own**: `ver.record` answers
`no_cpnp_path`, `ver.export` lists it in `skipped`. A raw's *exports* have Records, and those name it: by its
`content_hash` in `extensions["<authority>.lightcraft"].source` (not in `lineage`, whose parents must name a Record
that can be resolved, which a raw never has). A generated demo photo has no file at all (`no_asset`).

## Export sidecars

When an export writes an image and `ver.config {exportSidecars}` is on (the default), the exported file gets its own
Records beside it: `<name>.ver.json` (1.1) and `<name>.ver-1.0.json` (1.0). They are made from the **exported bytes**,
canonicalised, hashed and embedded on the export's own thread right after encoding, and written after the file once its
final path is known; they take part in the export's conflict policy with the file (Unique renames all three, Skip skips
all three) and are never written over a library original.

- The chain: `color_adjust` when the photo's settings differ from import's (always for a raw source), `crop`,
  `resize` when the output size differs from the cropped native size, `denoise` with an AI-denoised source, or
  `transcode` when none of those applies; then `export` (in 1.1 with the file's `uri`); then `decode`, `normalize`,
  `embed`. `acquisition` is `other` at the export's start: the rendition entered custody when LightCraft made it.
- 1.1 `lineage`: the source photo as `{producer_id, record_id, content_hash, pixel_hash: {value, cpnp_version},
  relation}`, `relation` `derived` (or `exported` when the rendition's pixels are the source's). `record_id` is the id
  `ver.record` gives the source's 1.1 Record. The source's `pixel_hash` comes from the digest lane when it has the
  photo, else the source is canonicalised once.
- No platform metadata: an export's Record asserts only what the exported file says, so an export that strips
  metadata does not leak it through its sidecar; with "no metadata" the Records do not name file locations either.
- Nothing here fails or blocks an export. When no Record can be made, the export's result has one line in `notes`
  (`No VER Record written: it needs the search-by-meaning model …`, or `… no CPNP-1 path …` for a DNG export).

## In the app

- **Settings ▸ Search ▸ VER Records**: whether Records can be made here (the model, the signing key), the authority,
  "Write VER Records beside exports", the signing key (shown by its key id) and the default schema (`ver.status`,
  `ver.config`). The fingerprint counts say how many photos are pixel-hashed (full-size Canonical Buffer) and how many
  provisional (raws, demo photos, the web build: thumbnail-level, not VER).
- **Photo ▸ Show VER Record…**: the photo's Record (conformance badge, level, schema, identity with Copy, space id,
  issues, JSON with Copy and Save…), or why it has none. It is made on the engine's VER thread (`ver.export
  {background}` into a private folder, then validated by `ver.import`), so the window never waits.
- **File ▸ Write VER Records…**: the selection (or every photo in view) into a folder via `ver.export {background:
  true}`, under the "Writing VER records" activity row; the toast gives the counts, and "Show skipped" lists each photo
  and why.
- **File ▸ Read VER Record…**: `ver.import` on a file, with status, summary, issues and matches grouped as Same file /
  Same pixels (a grouping, not identity) / Looks alike; Select Matches selects them.
- The export dialog's "Write VER Records" overrides `exportSidecars` for that export only, and the toast says once per
  batch how many files got no Record and why. Show Duplicates labels each duplicate's tier, with "Same pixels" (equal
  `pixel_hash`) between "Same file" and the fingerprint tiers.

## Commands

| Command | Params → result |
|---|---|
| `ver.record` | `{id?, schema?: "1.1-draft" \| "1.0.1", level?: L0..L3}` → `{record, issues, conformant, level, notes, schema}`; errors `no_asset`, `no_cpnp_path`, `needs_model`, `not_conformant`, `source`, `model`, `invalid` |
| `ver.export` | `{ids?, dir, schema?: "both" (default) \| "1.1-draft" \| "1.0.1", background?: false}` → `{written, files, skipped: [{id, code, reason}], dir}`: `<content_hash>.ver.json` (and `<content_hash>.ver-1.0.json` with `both`), under a "Writing VER records" activity row. With `background` → `{queued, skipped, dir, background}`: written one at a time on a thread of its own as `vera.pump` runs, none while paused (the pump's result gains `ver: {pending, written, failed}`) |
| `ver.import` | `{path? \| record?}` → `{status: accepted \| rejected, accepted, issues, matches: [{id, by, score}], recordSummary, pendingDigests}` |
| `ver.status` | `{}` → `{ready, reasons, authority, generator, producerId, defaultSchema, exportSidecars, signing, model, lastProduced, background, claims}` |
| `ver.config` | `{authority?, exportSidecars?, signKeyPath?, defaultSchema?}` (each nullable: back to the default) → the settings; saved in `prefs.json` → `ver` |

None is journaled: a Record is derived, never part of the catalog.

## Reading Records: the consumer's rules

`ver.import` validates a Record under **its own schema** (`ver_version` 1.0 → 1.0.1, 1.1 → 1.1-draft): schema, then
the VERA profile, then the core's own consistency rules (`LCVER-E2`: `identity.pixel_hash` and `image` must state the
same size, both being the Canonical Buffer's, errata E2). Any error-severity issue **rejects** it: the issues are returned and
nothing is matched. An accepted Record is matched against every library picture with a fingerprint:

| `by` | When | `score` |
|---|---|---|
| `content` | its `content_hash` is the SHA-256 of a library file: the same bytes (dedupe is this tier's job, E27) | 1 |
| `pixel_hash` | the same rendered pixels under CPNP-1. A **grouping**: distinct files that render identically share it, and E27 forbids reading it as an identity | 1 |
| `pdq` / `phash` | its perceptual digests within the same-picture bounds of `lightcraft_vera::thresholds` (PDQ ≤ 31 bits when the library picture's PDQ quality is ≥ 50, else pHash ≤ 6) | differing bits |
| `vector` | only when its `canonical_visual` embedding is in **this producer's** space with the same weights pin (E29), at or over `EMBED_NEAR` (0.90). The space has no reference set, so this is never a cross-producer claim | cosine |

Pictures not fingerprinted yet cannot match (`pendingDigests`). Reading a Record changes nothing in the library.

## The digest lane is VER

The VERA digest lane ([vera.md](vera.md) → *The digest lane*) fingerprints every file with a CPNP-1 path over its
**full-size Canonical Buffer** with `lightcraft_ver::hashing::identity`, and keeps its `pixel_hash` and VER
`content_hash` with the PDQ and pHash: the same values a Record carries, so `ver.import` matches against them directly
and `photo.duplicates` lists exact `pixel_hash` groups first (tier `pixels`). Files without a CPNP-1 path keep
provisional thumbnail-level digests (`vera.status` → `provisional`).

## The oracle

The upstream Python validator is the reference: `PYTHONPATH=…/vera-main/packages/ver-validator/src python -m
ver_validator.cli --schema 1.0.1|1.1-draft --format json FILE [--signature-key PUB.pem]`. The engine's tests run it on
the Records they make whenever it is installed (`VER_ORACLE_PY` and `VER_ORACLE_SRC` name the interpreter and the
source folder) and require `conformant` with no warning but `VER404` (no reference set, by design) and `VER308` (unsigned,
unless the test signs; then the oracle verifies the signature, and only `VER404` remains).

## What is not claimed

- **No reference set, no tolerance**: LightCraft's vectors are not interoperable with anyone else's, and nothing says
  they are (VER404 on every Record).
- **PROVISIONAL identity, by the standard's rule, not by our result**: LightCraft's decoder reproduces every digest of
  the published decoder corpus (21 of 21, `verd-pillow/1`; the evaluation is [ver-cpnp.md](ver-cpnp.md), kept at that
  count by `cargo run -q -p xtask -- ver corpus`), so it is the second toolchain E12 asks for. PROVISIONAL lifts only
  between implementations *evaluated against a published profile* (1.1 §8.3), and this evaluation is published by the
  specification's own party; cross-producer `pixel_hash` comparability is a claim the standards track makes when it
  accepts the evaluation, not one this app makes for itself.
- **No C2PA validation**: a present manifest is `unverifiable`, never `valid`.
- `weights_sha256` is comparable only within this producer (E29); `record_id` only within it (E16).

## Not done yet

- The interoperable space: `google/siglip2-base-patch16-256` as a second downloadable model, so Records can declare
  `org.verstandard.space.siglip2-base-p16-256/1.0` (which has a published reference set) and, once a tolerance is
  measured against that set, its `reference` block.
- A `joint_visual_text` space for the text tower, so accepted descriptions can travel as contextual embeddings (1.1).
- C2PA validation; COSE signatures; by-reference carriage of oversized segments (1.1 §10.1).
