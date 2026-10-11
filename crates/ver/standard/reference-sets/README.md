# VER reference sets — version 1

**`reference_set_version` `1` · three spaces · ten images · measured 2026-08-25**

Ten images, and — for each of the three embedding spaces this project ships —
the vector each one is expected to produce. That is all this is, and it is the
artifact VER 1.1 draft §8.2 makes REQUIRED without anyone having published one.

Draft §8.2, which this directory exists to satisfy:

> `set_manifest_sha256` — SHA-256 of the RFC 8785 JCS serialization of the
> reference-set manifest, binding the declared tolerances to an exact set of
> reference images and reference vectors. **It is REQUIRED**: a `reference`
> block *is* a conformance claim, and a claim bound only to a URI is a claim
> about whatever that URI serves today (§15.3).
>
> **Reference-set manifest.** The manifest names, for each reference item, the
> image's `content_hash`, its expected Canonical Buffer digest (`pixel_hash`),
> and the digest of the expected vector bytes. Publishing the manifest — not
> merely a URI — is what makes a tolerance a testable claim.

This directory is row **10** of
[`standards/ver/1.1-draft/RELEASE.md`](../ver/1.1-draft/RELEASE.md): *reference-set
manifests published for every space this project ships*. Read
["What this does and does not claim"](#what-this-does-and-does-not-claim) before
citing it — the manifests are real and measured, and the descriptor wiring that
would turn them into a conformance claim is not written yet.

## Status

**VER 1.1 is not ratified, and neither is §8.2.** These manifests are published
regardless: they are measurements of real checkpoints against committed bytes,
and nothing about them depends on the draft's fate. What is unratified is the
specification text that makes `set_manifest_sha256` a requirement — not the
digests below.

## Contents

```
standards/reference-sets/
├── README.md                     this file
├── images/                       10 PNGs, 174,626 bytes — the reference items
├── siglip2-base-p16-256/manifest.json
├── siglip-base-p16-256/manifest.json
└── clip-vit-b32/manifest.json
```

One directory per space, named for the space's last label with its version
dropped. `items[].name` resolves against `images/`, which is shared: the three
sets are the same ten images measured through three different checkpoints, so a
cosine that differs between two spaces differs because the spaces differ.

## The three sets

`set_manifest_sha256` is the SHA-256 of the RFC 8785 (JCS) canonical form of the
manifest's parsed JSON. It is recorded **here and never inside the manifest** —
a document cannot carry its own digest.

| space_id | model | dim | `set_manifest_sha256` |
| --- | --- | --- | --- |
| `org.verstandard.space.siglip2-base-p16-256/1.0` | `google/siglip2-base-patch16-256` | 768 | `38ea43bca2ee0067bb8cfae667863f2393d33fe239e654ca2166d7fa9244e35f` |
| `org.verstandard.space.siglip-base-p16-256/1.0` | `google/siglip-base-patch16-256` | 768 | `798c29a812335f78e3870eba836d0bc308a05995880c4e410cea2d1e8bf4132b` |
| `org.verstandard.space.clip-vit-b32/1.0` | `openai/clip-vit-base-patch32` | 512 | `fe610a09416d21188f4595ebb26cc0147e4b998ebcdfd6f50efcea8a0ed6ac38` |

These are exactly the three spaces `verd.embed.MODEL_CHAIN` can load — the §6.2
fallback chain, in its own order: primary, fallback, last resort. The third is a
different, smaller space whose vectors are **not** comparable with the first
two's, which is why it has a set of its own rather than a footnote in theirs.

Recompute all three from the repository root:

```bash
PYTHONPATH=services/verd/src .venv/bin/python - <<'PY'
import hashlib, json, pathlib, rfc8785
for path in sorted(pathlib.Path("standards/reference-sets").glob("*/manifest.json")):
    document = json.loads(path.read_text(encoding="utf-8"))
    print(hashlib.sha256(rfc8785.dumps(document)).hexdigest(), document["space_id"])
PY
```

## What a manifest carries

```json
{
  "reference_set_version": "1",
  "space_id": "org.verstandard.space.clip-vit-b32/1.0",
  "items": [
    {
      "name": "01-gradient-scene.png",
      "content_sha256": "163dc8e9…",
      "pixel_sha256": "bd175fc1…",
      "vector_sha256": "2537406f…"
    }
  ]
}
```

(That is `clip-vit-b32/manifest.json`'s first entry with each digest elided
after eight characters. The prefixes are the real ones — an example digest in a
document about not shipping example digests would be a poor joke.)

- `content_sha256` — SHA-256 of the committed file's bytes, which is VER's
  `identity.content_hash` for that Asset.
- `pixel_sha256` — the Canonical Buffer digest, `verd.hashing.pixel_hash` over
  `verd.cpnp.canonicalize` of those same bytes: `SHA-256("VER1:" + width + ":" +
  height + ":sRGB8:" ‖ raw)` per errata E10. Every reference item is an
  untagged RGB PNG, so CPNP-1 decodes, composites nothing, converts nothing and
  resizes nothing — the Canonical Buffer is the PNG's own pixels.
- `vector_sha256` — SHA-256 of the packed vector octets defined below.

Draft §8.2 names these three facts per item in prose and defines no JSON Schema
for the manifest — only the outer `set_manifest_sha256` is schema-encoded. The
shape above is the one `RELEASE.md`'s worked example uses, and this directory
adopts it unchanged so that the two do not describe different documents.

The `items[]` array is sorted bytewise-ascending by `name`, and the numeric
filename prefix makes that order the documentation order too.

## The images

Ten PNGs, each drawn by closed-form integer arithmetic over the pixel grid —
no seeded RNG, no fonts, no floating point. `--check` regenerates them and
byte-compares, so the committed bytes are provably the ones the generator
draws. They are chosen to spread across the failure modes an image-space
preprocessing pipeline actually has, not to be pretty.

| name | Canonical Buffer | bytes | what it is for |
| --- | --- | --- | --- |
| `01-gradient-scene.png` | 192×144 | 4,322 | the busy asymmetric scene the CPNP suites use: broad low-frequency structure |
| `02-text-document.png` | 200×256 | 742 | document-shaped: high-contrast thin dark bars on a light page, portrait |
| `03-high-frequency-texture.png` | 160×160 | 59,754 | detail at the grid's Nyquist limit — the sharpest test of how a resize interpolates |
| `04-low-contrast-field.png` | 176×132 | 2,481 | all content inside a 20-level band around mid-grey |
| `05-saturated-primaries.png` | 192×128 | 339 | the corners of the sRGB cube, plus one asymmetric marker |
| `06-near-black.png` | 144×108 | 1,554 | structure inside the bottom 5% of the range |
| `07-near-white.png` | 144×108 | 1,560 | the same, 12 levels below saturation |
| `08-diagonal-composition.png` | 208×128 | 3,247 | non-square, off-centre subject, hard diagonal edge — a centre crop is visible |
| `09-radial-pattern.png` | 176×176 | 26,846 | concentric rings about an off-centre origin, spacing narrowing outwards |
| `10-photographic-composite.png` | 256×192 | 73,781 | photograph-like statistics: sky gradient, sun, ridge silhouette, grain, vignette |

No two items are near-duplicates in any of the three spaces: the largest cosine
between two distinct items is **0.916** (siglip2), **0.902** (siglip) and
**0.935** (clip). A reference set whose items sat on top of each other would let
a tolerance be met by a model that had learned nothing.

## Measurement block

Every vector digest came out of a real forward pass through `verd.embed`'s own
code path — `_load_from_chain`, `_probe_dim`, `_to_pil`, `_forward`,
`l2_normalize` — against a checkpoint resolved from the local HuggingFace cache.
`verd.embed.MODEL_CHAIN` was narrowed to one entry per measurement, because the
chain is a preference order for picking *one* space to run and a reference set
needs each space pinned in turn; nothing else about the load path was changed.

| | |
| --- | --- |
| measured | 2026-08-25 |
| device | `cpu` (`MODEL_DEVICE=cpu`; `verd.embed.get_device()` asserted, not assumed) |
| `torch` | 2.13.0+cpu |
| `transformers` | 5.15.1 |
| `numpy` | 2.4.6 |
| `Pillow` | 12.3.0 (image generation and CPNP-1 decode) |
| intra-op threads | `torch.set_num_threads(1)`, pinned — see below |
| batching | one image per `embed_images()` call |

Per space, as measured (`dim` from a real forward pass via `_probe_dim`, never a
constant; `weights_sha256` from `verd.space_cache.weights_sha256`, which digests
the checkpoint files actually on disk):

| space | dim | `weights_sha256` | resize | interpolation | centre crop | mean/std |
| --- | --- | --- | --- | --- | --- | --- |
| siglip2-base-p16-256 | 768 | `95b489cffc984428c3fd6f582c7c974fc6d0118e3974c2af0459ca305b313bfa` | yes | bilinear | no | 0.5 / 0.5 |
| siglip-base-p16-256 | 768 | `d4cac05964418bd95ff4a28edb05cb42b80d7f6607d0b95a45fdc890d952c1e1` | yes | bicubic | no | 0.5 / 0.5 |
| clip-vit-b32 | 512 | `0d2467ba23703cb2df01ad1eb1cd8c92688b1971f869d7dec850e8b8b79a646c` | yes | bicubic | yes | CLIP's own |

`weights_sha256` is producer-scoped by errata **E29** — it has no defined
preimage in VER 1.x, so these values are comparable only within this producer.
The interoperable pin is ADR-0007's bundle manifest, which this project does not
yet emit.

The two SigLIP spaces share a dimension and differ in interpolation alone among
the members above; their vectors are still different vectors, and the manifests
say so item by item.

## Vector packing

`vector_sha256` is `SHA-256` over exactly these octets:

1. Take the space's image embedding as `verd.embed.embed_images` returns it:
   `float32`, **L2-normalized** (`normalization: "l2"` in the descriptor).
2. Cast to the descriptor's own `dtype`, `fp16` — IEEE 754 binary16.
3. Serialize **little-endian**, component 0 first, no header, no padding:
   `dim × 2` octets. 1536 for the two SigLIP spaces, 1024 for CLIP.

```python
np.ascontiguousarray(np.asarray(vector, dtype=np.float32).astype("<f2")).tobytes()
```

Normalization happens in `float32` and the cast follows it, which is the order
`verd` stores vectors in. The consequence is that the packed vector is **not
exactly** unit-norm: across these thirty vectors the binary16 norms lie in
[0.99980, 1.00018]. Anything comparing them must normalize after unpacking, and
the cosine metric the spaces declare does that anyway.

## What a vector digest is and is not

A `vector_sha256` **binds the set**. It is what stops a published tolerance from
silently coming to refer to a different set of vectors than the one it was
measured against — the §15.3 defect the whole `set_manifest_sha256` requirement
exists to close.

It is **not** the conformance test, and this measurement shows why. Holding the
checkpoint, the toolchain, the device and the images fixed, and changing only
how the forward pass was scheduled:

| change | digests still matching (of 10) | worst cosine |
| --- | --- | --- |
| | siglip2 · siglip · clip | |
| a fresh process, same pins | 10 · 10 · 10 | 1.0 |
| all ten images in one `embed_images()` call (CPU batches of 8 + 2) | 3 · 4 · 3 | ≥ 0.9999999 |
| `torch.set_num_threads(4)` instead of 1 | 0 · 0 · 6 | ≥ 0.9999999 |

The vectors are the same vectors — the disagreements are last-place bits that
survive the binary16 cast — but the digests are not the same digests. Two
consequences, both deliberate:

- **The published construction is pinned to what a reader can reproduce.**
  Threads are set to 1 rather than left at whatever the measuring machine had,
  and each image is embedded in a call of its own so that reproducing item 7
  requires item 7 and nothing else. A digest that only reproduces when nine
  other images are in the same batch is not a digest anyone can check.
- **Conformance is the declared cosine tolerance, not digest equality.** Draft
  §8.2 sets the float floor at ≥ 0.99 and says descriptors SHOULD declare ≥
  0.999; the largest scheduling difference above is a cosine deficit of about
  6 × 10⁻⁸, four to five orders of magnitude inside either bar. A specification
  that asked for bit-exact vectors would
  be asking for something no two BLAS builds deliver, which is precisely the
  distinction between this and the decoder corpus next door: CPNP-1 is
  integer-exact and `standards/decoder-corpus/1/` really does demand every
  digest exactly (errata E12), while a forward pass in floating point does not
  and must not.

Bit-exact reproduction of these digests therefore requires the pinned toolchain
*and* the same CPU kernel dispatch. Cross-architecture bit-exactness has not
been measured and is not claimed.

## What this does and does not claim

**Claimed on every commit**, by `tests/test_record_refset_manifests.py` in a
tier with no torch and no network:

- Three manifests whose every `content_sha256` and `pixel_sha256` recomputes
  from the committed image bytes.
- Three `set_manifest_sha256` values that recompute from the manifests and are
  found in this file.
- The three `space_id` values, against `verd.space_cache.SPACE_IDS`, so a space
  the pipeline can load with no set published for it fails the suite.

**Claimed by `--check`**, which regenerates: ten committed images whose bytes are
exactly what the closed-form generator draws.

**Claimed by measurement, and re-checkable on demand**: thirty `vector_sha256`
values, each the digest of a vector that was actually produced by the named
checkpoint. Not one is illustrative — errata **E26** is the item that catalogues
shipping placeholder digests as though they were real, and a reference set of
invented vectors would repeat that defect at the one place §8.2 makes
load-bearing. CI cannot afford ~1.5 GB of weights, so it re-checks none of them;
`--check --vectors` re-checks all thirty, and the test suite re-checks three
(one per space) wherever torch is installed.

**Not claimed.** These manifests make `set_manifest_sha256` *satisfiable*. They
do not by themselves make a conformance claim, because a conformance claim is
made by a **space descriptor**, not by a manifest:

- §8.2's tolerances — `tolerance_cosine`, `tolerance_cosine_p5`,
  `tolerance_cosine_worst`, `tolerance_hamming_max` — are **declared by a
  `spaceDescriptor.reference` block**. Nothing in this directory declares one,
  and nothing in this directory can: a tolerance is a producer's assertion about
  its own reproducibility, and the manifest is the evidence that assertion is
  measured against.
- **This project's pipeline does not publish reference-bearing descriptors.**
  `verd.models.SpaceDescriptor.to_json()` has no `reference` key at all, so
  every Record `verd` currently emits warns `VER404`
  (`space-reference-missing`), and errata **E13** rates such a descriptor as
  making no conformance claim. Wiring `reference` into the descriptor —
  `set_uri`, the `set_manifest_sha256` values above, and tolerances measured by
  re-embedding this set across the runs a producer wants to claim
  reproducibility over — is **follow-up work that has not been done**, not a
  property this directory confers.
- No tolerance number appears anywhere in this directory, deliberately. The
  cosine figures above are *sensitivity measurements of one implementation
  against itself* over thirty vectors. A `tolerance_cosine` is a floor a
  producer is prepared to be held to across checkpoint reloads, library
  upgrades and hardware — an empirical corpus `RELEASE.md` lists as its own
  outstanding ratification prerequisite. Publishing 0.9999999 as a tolerance on
  the strength of this run would be exactly the "three round figures with a
  coherence argument and no data" the draft criticises.

## Reproducing this directory

The generator is
[`scripts/make_refset_manifests.py`](../../scripts/make_refset_manifests.py).
The committed bytes are the reference set; the generator is subordinate to them.

```bash
# verify images, content/pixel digests and the set_manifest_sha256 values above.
# Torch-free. This is the stronger of the two torch-free checks: it *regenerates*
# the images and byte-compares, which the CI suite does not do.
PYTHONPATH=services/verd/src .venv/bin/python scripts/make_refset_manifests.py --check

# ... and recompute all thirty vector digests too. Needs torch, transformers and
# the three checkpoints; MODEL_DEVICE=cpu, HF_HOME pointing at a warm cache.
MODEL_DEVICE=cpu PYTHONPATH=services/verd/src python scripts/make_refset_manifests.py --check --vectors

# every digest here except the vectors, recomputed from the committed bytes.
# This is what the standards CI tier runs; it installs no torch, so the three
# vector cases skip.
PYTHONPATH=services/verd/src:packages/ver-validator/src \
  .venv/bin/python -m pytest tests/test_record_refset_manifests.py -q
```

A `--check` failure is never repaired by regenerating this directory in place.
Implementations will by then have been evaluated against the published digests,
and moving them quietly retracts a conformance claim without saying so:
corrections ship as `standards/reference-sets/` version 2, with
`reference_set_version` incremented and both versions served.
