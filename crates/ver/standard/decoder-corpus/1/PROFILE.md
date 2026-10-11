# Decoder implementation profile `verd-pillow/1`

**Corpus release 1 · corpus id `ver-decoder-corpus/1` · 2026-08-25**

This document is the first published **decoder implementation profile** for
CPNP-1, and the corpus beside it is the first published **reference corpus**.
Together they are what errata item **E12** requires and what VER 1.0 shipped
without.

E12, in full, on what a conformant decoder is:

> A decoder is **conformant** for CPNP-1 with respect to a published
> **implementation profile**: a named, versioned pinning of the decode
> toolchain (library and version, build options that affect pixel output,
> chroma upsampling and IDCT selections) together with a **reference corpus**
> of Assets and their expected Canonical Buffer digests. An implementation is
> conformant for that profile iff it reproduces every expected digest in the
> corpus exactly.

That is the whole conformance rule. There is no partial credit, no tolerance,
and no second criterion: **reproduce every digest in `manifest.json` exactly,
or you are not conformant for `verd-pillow/1`.**

---

## 1. What this profile names

| | |
| --- | --- |
| Profile identifier | `verd-pillow/1` |
| Corpus identifier | `ver-decoder-corpus/1` |
| Procedure pinned | CPNP-1 (`cpnp_version` `"1.0"`) |
| Reference implementation | `services/verd/src/verd/cpnp.py` + `services/verd/src/verd/hashing.py` |
| Corpus | `corpus/` — 21 Assets, 145,311 bytes |
| Expected digests | `manifest.json` |
| Conformance run | `tests/test_record_decoder_corpus.py` |

**Three version planes, none of which is the other.** `cpnp_version` is `"1.0"`
and stays `"1.0"` for the whole VER 1.x line — any change to CPNP is CPNP-2 and
therefore VER 2.0 (`VERSIONING.md` §4). `ver_version` is the record format.
The trailing `1` in `verd-pillow/1` and `ver-decoder-corpus/1` is a **third**
plane: the version of *this profile and its corpus*, which moves when the
toolchain pins or the corpus contents change, without either of the other two
moving. A new Pillow release does not change CPNP-1; it changes which
implementations are conformant for `verd-pillow/1`, which is exactly the
distinction E12 exists to draw.

**This release directory is immutable once published.** Corrections, additions
and toolchain bumps ship as `standards/decoder-corpus/2/` with a new profile
identifier. Editing a published digest would silently retract every conformance
claim made against it — the same reasoning that freezes `standards/ver/1.0.0/`
with its defects (`VERSIONING.md` §3, `CLAUDE.md`).

---

## 2. The pinned toolchain

Recorded from the build that produced the committed corpus bytes and the
digests in `manifest.json`.

| Component | Pinned value | What it decides |
| --- | --- | --- |
| Pillow | **12.3.0** | Decoding for every container; the orientation transposes; palette expansion; `Image.alpha_composite`; mode conversion at quantisation |
| libjpeg | API level **6.2**, as bundled in the Pillow 12.3.0 wheel (libjpeg-turbo) | **IDCT precision and chroma upsampling** — the two disagreements E12 names by name |
| libwebp | **1.6.0** | WebP lossless decode |
| libtiff | **4.7.1** | TIFF strip layout and LZW decode |
| zlib | **1.3** | PNG inflate |
| littleCMS | **2.19**, reached through `PIL.ImageCms` | The ICC → sRGB transform: intent handling, black point compensation, rounding |
| NumPy | **2.4.6** | The `uint8` `[H, W, 3]` C-contiguous buffer whose `tobytes()` is hashed |
| Python | **3.11.15** (CPython) | Recorded for completeness; nothing in the decode path depends on it |

**Other Pillow versions must be evaluated, not assumed.** Nothing in this
profile predicts the behaviour of Pillow 12.4, 13.0, a system-libjpeg build, a
Pillow-SIMD build, or a distribution package that links different codecs. Each
is a *candidate implementation* under E12: run the corpus, and if every digest
reproduces, that build is conformant for `verd-pillow/1` too. If any digest
moves, it is not, and no amount of version-number proximity substitutes for the
run. This is the practical content of E12 — "conformant decoder" names a test,
not a dependency range (`VERSIONING.md` §3.4, condition 3).

Pillow does not expose libjpeg-turbo's own version string through
`PIL.features`; `features.version("jpg")` reports the libjpeg **API** level
(`6.2`), not the turbo release. An implementation that wants to match this
profile by construction rather than by evaluation should pin the Pillow *wheel*,
which carries the codec builds with it.

---

## 3. CPNP-1 as this profile executes it

Every selection below is made by `services/verd/src/verd/cpnp.py`. They are
listed because a digest is only reproducible by someone who knows which choices
produced it.

### 3.1 Decode

Pillow's plugin registry decides what is an image. There is no format
allowlist at the CPNP layer: whatever Pillow opens and `load()`s is decoded,
and `img.format.lower()` is recorded as `image_format`. Anything Pillow refuses
— empty input, a truncated header, a text file — raises `CpnpError` and **no
Record is produced for it**. Undecodable inputs are rejected, per §4 step 1.

### 3.2 Frame selection, and what "the first frame" means

Multi-frame inputs are decoded at **logical index 0** (`seek(0)`) — never
averaged, never the "best" frame, never advanced past. The `animated` flag is
set only when the format is one of `gif`, `webp`, `png`, `apng`, `avif` *and*
`n_frames > 1`; a multi-page TIFF is therefore paged, not animated, and still
decodes to page 0.

Errata **E7** defines "the first frame" for containers where a frame is not the
whole picture, and this profile executes all three of its rules:

1. **Fully coalesced.** Frame 0 is composed onto the container's logical screen
   at the container's full logical dimensions. **The Canonical Buffer's
   dimensions are the logical screen's, never the frame rectangle's.**
2. **The canvas beneath frame 0 is transparent.** Every logical-screen pixel
   frame 0's rectangle does not cover, and every pixel frame 0 marks
   transparent, is `RGBA = (0, 0, 0, 0)` entering §3.5 and is therefore **opaque
   white** in the Canonical Buffer. A container-declared background colour is
   **not** consumed: GIF89a's Logical Screen Descriptor Background Color Index
   is ignored, as is the Background Color field of a WebP `ANIM` chunk.
3. **Disposal never applies.** Disposal describes how a frame is cleared before
   the *next* frame is drawn; CPNP-1 never advances past index 0, so no disposal
   method of any frame is consumed.

Corpus entries:

- `corpus/gif-animated-2frame.gif` — two full-screen frames that differ in every
  pixel, so a decoder that took frame 1 cannot reproduce the digest.
- `corpus/gif-frame0-subrectangle.gif` — a two-frame GIF whose frame 0 is a
  40×30 rectangle at offset (12, 9) over a 64×48 logical screen, carrying
  disposal method 2 (restore-to-background) and a Logical Screen Descriptor
  background index pointing at a colour table entry that holds magenta
  `(255, 0, 255)` and that no pixel of either frame references. Its expected
  Canonical Buffer is 64×48, its uncovered region is opaque white, and magenta
  appears nowhere in it. A decoder that hashed the frame rectangle fails on the
  dimensions; one that painted the container background, or that pre-applied
  frame 0's disposal, fails on the pixels.

**One E7 case this profile does not reach, stated because it is a divergence
and not a gap in coverage.** Where a GIF's frame 0 is a sub-rectangle and the
file declares *no* transparency index, this toolchain renders the uncovered
region as palette index 0 rather than as E7's transparent canvas, so the
Canonical Buffer carries that palette colour where E7 requires opaque white. No
corpus entry pins that case: a reference corpus must not publish an expected
digest for behaviour a normative errata item forbids. It is recorded in §6 and
needs a ruling — an implementation fix, or an errata amendment — before release
2 can carry a digest for it either way.

### 3.3 Orientation — an explicit map, not `ImageOps.exif_transpose`

EXIF tag **274** is read from `img.getexif()` and applied by a map that lives in
`cpnp.py`. `ImageOps.exif_transpose` would do the same thing today and possibly
something else after an upstream refactor; a hash that depends on a library
default is not a standard.

| Tag 274 | Operation applied, in order | Canonical Buffer dimensions |
| --- | --- | --- |
| 1 | *(none — the no-op)* | `W × H` |
| 2 | `FLIP_LEFT_RIGHT` | `W × H` |
| 3 | `ROTATE_180` | `W × H` |
| 4 | `FLIP_TOP_BOTTOM` | `W × H` |
| 5 | `FLIP_LEFT_RIGHT`, then `ROTATE_90` | `H × W` |
| 6 | `ROTATE_270` | `H × W` |
| 7 | `FLIP_LEFT_RIGHT`, then `ROTATE_270` | `H × W` |
| 8 | `ROTATE_90` | `H × W` |

`Image.Transpose.ROTATE_90` is counter-clockwise, which is why orientation 6
(the common "camera rotated 90° clockwise") maps to `ROTATE_270`. Values 5 and 7
are written as mirror-then-rotate rather than `TRANSPOSE`/`TRANSVERSE`; they are
numerically identical, and the pair is spelled out so the intent is readable.

**Out of range is upright.** Any value that is not 1–8 — `0`, `99`, a negative,
a string, a corrupt IFD that raises on read — is logged and treated as **1**.
It is not an error, and it does not reject the asset.

Corpus entries: `corpus/jpeg-orientation-1.jpg` … `-8.jpg` (one scene, one
encoder setting, differing in tag 274 and nothing else) and
`corpus/jpeg-orientation-out-of-range.jpg` (tag 274 = 99), whose expected digest
equals orientation 1's.

**Orientation is render-affecting metadata.** Removing it *moves* `pixel_hash`,
and the corpus proves it: all eight orientation entries carry distinct digests.
This is errata item **E1** / finding `THESIS-01`, and it is the reason the
thesis is stated as "`pixel_hash` survives stripping of **descriptive**
metadata" rather than of metadata in general. `corpus/jpeg-no-exif.jpg` is the
other half: the same picture with the entire EXIF block never written, sharing
orientation 1's digest exactly.

### 3.4 Colour — ICC → sRGB, fail-closed

An asset carrying an embedded ICC profile is converted to sRGB through
littleCMS:

```
ImageCms.profileToProfile(
    image,
    ImageCms.ImageCmsProfile(BytesIO(icc_profile)),   # source: the embedded profile
    ImageCms.createProfile("sRGB"),                   # destination: the lcms built-in sRGB
    renderingIntent = Intent.RELATIVE_COLORIMETRIC,   # = 1
    outputMode      = "RGB",
    flags           = Flags.BLACKPOINTCOMPENSATION,   # = 0x2000 (8192)
)
```

- **Rendering intent: relative colorimetric** (`1`).
- **Black point compensation: ON.** The `BLACKPOINTCOMPENSATION` flag (`0x2000`)
  is passed, and no other flag is. This is stated flatly because it is easy to
  get backwards, and a profile document that named the wrong flag would send
  every implementer to a digest they cannot reproduce.
- **Destination profile:** littleCMS's *built-in* sRGB
  (`ImageCms.createProfile("sRGB")`), not a shipped sRGB binary. Note that
  lcms stamps a synthesised profile with the wall-clock creation date, so the
  destination profile's own bytes are not constant between runs — only its
  colorimetry is, and only its colorimetry is used.
- **Input mode:** modes littleCMS consumes directly (`RGB`, `L`, `CMYK`, `LAB`)
  are handed over as-is; anything else is promoted to `RGB` first.
- **An absent profile is assumed sRGB** and converted by nobody. Absent is not
  broken.
- **A present-but-unusable profile fails closed.** If the profile cannot be
  parsed, or no transform to sRGB can be built from it, `CpnpError` is raised
  and **no Canonical Buffer, and therefore no Record, exists for that asset**.
  Errata **E8**: the untagged-sRGB fallback applies to assets that declare no
  colour space, never to assets that declare one that cannot be read. Treating
  an unusable profile as untagged sRGB reinterprets every colour in an Adobe RGB
  or CMYK-derived image and yields a `pixel_hash` no other implementation
  reproduces — a false identity, not a weaker one. There is deliberately no
  field in which a Record could confess this happened.

Corpus entry: `corpus/png-srgb-icc.png`. Under this toolchain the sRGB → sRGB
transform is **exactly identity on 8-bit input** — measured, not assumed — so
that entry shares its digest with the untagged PNG, the WebP and the two TIFFs
of the same picture. A CMM that rounded that transform differently would fail
this corpus, which is the point of pinning one.

### 3.5 Alpha — set aside, then composited over white

Alpha is detected as *either* a real alpha channel (`"A" in img.getbands()`)
*or* palette/colour-key transparency (`"transparency" in img.info`). Both count.

The channel is extracted **before** colour management, because littleCMS does
not carry alpha through a transform unless asked and a silently zeroed alpha
would composite the whole image to white. After the sRGB conversion the alpha is
put back and the image is composited over **opaque white** `(255, 255, 255, 255)`
**in the gamma domain** — not in linear light. That is not photometrically
correct; it is what every other implementation does, and agreement beats
correctness when the output is an identity.

Corpus entries: `corpus/png-rgba-alpha.png` (a diagonal alpha ramp: one corner
fully transparent and therefore pure white in the buffer, the opposite corner
fully opaque and therefore untouched) and
`corpus/gif-palette-transparency.gif` (a palette colour key, not an alpha
channel, honoured all the same).

### 3.6 Quantisation — 8 bits per channel, always

The buffer is `RGB`, 8 bits per channel. Sources deeper than 8 bits
(`I`, `I;16`, `I;16B`, `I;16L`, `I;16N`, `F`) are quantised down with a warning,
because quantisation changes what the pixel hash means. The final buffer is
always `[H, W, 3]` `uint8`, C-contiguous; anything else raises.

### 3.7 No resize. No crop. Ever

The Canonical Buffer is the decoded image at its own dimensions. Resizing is the
model processor's business; if it happened here, `pixel_hash` would encode the
input resolution of whichever model version was fashionable that quarter.
Orientation may *transpose* the dimensions (§3.3); nothing else changes them.

### 3.8 HDR gain maps are flagged, never tone-mapped

The raw bytes are scanned for `urn:iso:std:iso:ts:21496:-1`, `hdrgm:Version` and
`GainMapImage`. A match sets `hdr_present`. Pillow hands back the SDR base
image and CPNP-1 hashes that; no tone mapping happens. *(No corpus entry — see
§6.)*

### 3.9 The digest itself

`pixel_hash` is SHA-256 over this exact preimage, rendered as 64 lowercase hex
characters:

```
b"VER1:" + ascii(width) + b":" + ascii(height) + b":sRGB8:" + <row-major interleaved RGB8 bytes>
```

Per errata **E10**: `width` and `height` are decimal ASCII with no leading
zeros, no sign, no separators and no whitespace; the trailing bytes are exactly
`width · height · 3` octets, interleaved R, G, B, top-to-bottom, row-major, with
no row padding and no alpha channel — alpha has already been consumed by §3.5.
The literal `VER1:` denotes **CPNP-1**, and a Record whose `pixel_hash` was
computed with it MUST carry `cpnp_version: "1.0"`; changing the tag is CPNP-2,
which is VER 2.0 (`VERSIONING.md` §4). `content_sha256` in the manifest is the
far simpler SHA-256 over the committed file's bytes, taken before anything
touches them.

The perceptual digests (`pdq/1.0`, `phash-dct/64`) are **not** part of this
profile. They are computed *from* the Canonical Buffer, so this corpus is a
prerequisite for pinning them, but their own parameters and test vectors are
errata **E22**'s deferred work for the 1.1 perceptual-algorithm registry.

---

## 4. PROVISIONAL, and how an implementation stops being it

E12's second paragraph, in full:

> An implementation that has not been evaluated against a published profile MAY
> produce Records, but those Records' identity claims are **PROVISIONAL**: the
> Producer MUST NOT advertise cross-producer `pixel_hash` comparability for
> them, and a Consumer MUST NOT treat `pixel_hash` equality across unevaluated
> implementations as render identity.

So PROVISIONAL is not a defect and not a lower conformance level. It is a
statement about *what a `pixel_hash` from that implementation is evidence of*:
it remains a perfectly good within-producer grouping key, and it is not evidence
of render identity against anybody else's records.

E12's third paragraph said this:

> Publication of the first reference corpus and implementation profile is a VER
> 1.1 deliverable. Until it exists, every implementation is PROVISIONAL in this
> sense, and this annex says so rather than implying otherwise.

**That sentence is now discharged.** This directory is the first published
reference corpus and implementation profile. From its publication:

- An implementation that runs this corpus and reproduces **every** digest in
  `manifest.json` exactly is **conformant for `verd-pillow/1`**, and may drop
  PROVISIONAL for the Records it produces under it — that is, it may advertise
  cross-producer `pixel_hash` comparability **with other implementations
  conformant for the same profile**, and a Consumer may treat `pixel_hash`
  equality between two such implementations as render identity.
- An implementation that has not run the corpus, or that runs it and misses any
  digest, stays PROVISIONAL. Nothing about it has got worse; it simply has not
  made the claim.
- Comparability is **scoped to the profile**. Conformance for `verd-pillow/1`
  says nothing about an implementation conformant for some future
  `verd-pillow/2` or a differently-pinned profile. Two profiles are comparable
  only where a published evaluation says they agree, and no such evaluation
  exists yet.
- A Record does not carry a profile identifier — VER 1.0.x and the 1.1 draft
  have no field for one. A Producer states its profile **outside** the Record,
  the way E29 requires a Producer to document what its `weights_sha256` covers.

---

## 5. What this establishes, and what it does not

**Establishes.** `VERSIONING.md` §3.4 condition 3 — "a published decoder corpus
exists for anything the release makes a pixel-level claim about (E12), so that
'conformant decoder' names a test rather than an intention" — now has an
artifact to point at, and `standards/ver/1.1-draft/RELEASE.md` row 9 has a
deliverable. The corpus is executed on every commit by
`tests/test_record_decoder_corpus.py`, so the claim is checked rather than
asserted.

**Does not establish.** *One* implementation has been evaluated against this
corpus: the one that generated it. A corpus that only its author has run
demonstrates determinism and self-consistency, not interoperability. Until a
second, independently built toolchain reproduces every digest here, the
cross-producer comparability that dropping PROVISIONAL unlocks has an evaluated
population of one. That second evaluation is `RELEASE.md` row 14 —
`VERSIONING.md` §3.4 conditions 1 and 2 — and it remains outstanding. Publishing
this profile does not close it, and this document does not pretend otherwise.

---

## 6. What corpus release 1 does not pin

Stated plainly, because a reference corpus that quietly omits a code path is
worse than one that names its gaps.

| Not covered | Why, and where it should land |
| --- | --- |
| **Non-sRGB ICC sources** (Adobe RGB, Display P3, CMYK, Lab) | The single ICC entry is sRGB-tagged, where the transform measures as exact identity — so §3.4's intent and BPC flag are *declared* by this profile but not *exercised* by any digest in release 1. This is the highest-value addition for release 2. None ships here because no non-sRGB profile can be synthesised from the pinned toolchain alone (`ImageCms.createProfile` offers sRGB, LAB and XYZ, and an lcms LAB profile will not build a transform under §3.4's settings), and shipping a third-party binary profile raises a licensing question this release does not settle |
| **The fail-closed rule** (§3.4, E8) | An unusable profile produces no Canonical Buffer and therefore no digest; `manifest.json`'s shape has no row for "must raise". Covered instead by `tests/test_cpnp_determinism.py::test_broken_icc_profile_fails_closed` |
| **A sub-rectangle frame 0 with no declared transparency index** (§3.2) | **A known divergence, not merely uncovered.** This toolchain renders the uncovered region of the logical screen as palette index 0; E7 requires the canvas to be transparent and therefore opaque white. Reproduced against `services/verd/src/verd/cpnp.py` at the time of this release. Publishing a digest for it would publish a normative violation, so release 1 does not. Needs a ruling before release 2 |
| **Sources deeper than 8 bits** (16-bit PNG/TIFF, float TIFF) | §3.6's quantisation is unexercised. Pillow's `I;16 → RGB` conversion clips rather than scales, so an entry here would pin behaviour that deserves an errata item first |
| **HDR gain maps** (§3.8), **AVIF**, **HEIC** | AVIF decodes in this build (libavif 1.4.2) and HEIC needs a plugin that is not installed. Neither is in `ingest.py`'s directory-walk allowlist today |
| **Progressive, arithmetic-coded, and CMYK/APP14 JPEG** | All three are real-world inputs whose decode differs meaningfully between libraries; all three belong in release 2 |
| **Animated WebP and APNG** | §3.2's frame-0 rule is exercised for GIF only |
| **Very large images** | Out of scope for a decode profile; §7's size limits are `RELEASE.md` row 15's separate measured-corpus deliverable |

---

## 7. Provenance of this directory

Every image was produced by `scripts/make_decoder_corpus.py` from closed-form
integer arithmetic over the pixel grid — no seeded RNG, no fonts, no floating
point. Every digest in `manifest.json` was computed from the committed bytes by
`verd.cpnp.canonicalize` and `verd.hashing.pixel_hash`; none was written by
hand. `--check` regenerates the whole tree into a temporary directory and
byte-compares, so the claim "these are the bytes that script produces" is itself
executable:

```
.venv/bin/python scripts/make_decoder_corpus.py --check
```

If a future encoder emits different bytes for the same drawing, that command
fails and the answer is a new release directory — never an in-place
regeneration of this one.

## 8. References

- `docs/standards/VER-1.0-errata.md` — the errata ledger (E1, E8, E10, E12, E22)
- `standards/ver/1.0.1/VER-1.0.1-errata-annex.md` — the normative item text
- `standards/ver/1.1-draft/VER-1.1-draft.md` §8.3 — the corpus as a 1.1 deliverable
- `standards/ver/1.1-draft/RELEASE.md` row 9 — the blocking checklist row
- `VERSIONING.md` §3.4 condition 3 — the publication bar this satisfies; §4 — the CPNP invariant
- `services/verd/src/verd/cpnp.py`, `services/verd/src/verd/hashing.py` — the reference implementation
