# VER decoder reference corpus — release 1

**Corpus id `ver-decoder-corpus/1` · profile `verd-pillow/1` · 2026-08-25**

Twenty-one small images and the Canonical Buffer digest each one must produce.
That is all this is, and it is the artifact VER 1.0 shipped without.

Errata item **E12**, which this directory exists to satisfy:

> A decoder is **conformant** for CPNP-1 with respect to a published
> **implementation profile**: a named, versioned pinning of the decode
> toolchain (library and version, build options that affect pixel output,
> chroma upsampling and IDCT selections) together with a **reference corpus**
> of Assets and their expected Canonical Buffer digests. An implementation is
> conformant for that profile iff it reproduces every expected digest in the
> corpus exactly.
>
> An implementation that has not been evaluated against a published profile MAY
> produce Records, but those Records' identity claims are **PROVISIONAL**: the
> Producer MUST NOT advertise cross-producer `pixel_hash` comparability for
> them, and a Consumer MUST NOT treat `pixel_hash` equality across unevaluated
> implementations as render identity.
>
> Publication of the first reference corpus and implementation profile is a VER
> 1.1 deliverable.

Read [`PROFILE.md`](PROFILE.md) for the toolchain pins and for every CPNP-1
behavioural selection the digests below depend on. This file is the operating
manual.

## Contents

```
standards/decoder-corpus/1/
├── PROFILE.md      the implementation profile: toolchain pins + CPNP-1 selections
├── README.md       this file
├── manifest.json   the expected digests — the only place any digest appears
└── corpus/         21 Assets, 145,311 bytes
```

`manifest.json`:

```json
{
  "corpus": "ver-decoder-corpus/1",
  "profile": "verd-pillow/1",
  "files": [
    { "path": "corpus/…", "content_sha256": "…", "width": 0, "height": 0, "pixel_hash": "…" }
  ]
}
```

- `path` — relative to this directory. Entries are sorted bytewise-ascending by
  path and are unique.
- `content_sha256` — SHA-256 over the committed file's bytes. Check it first: if
  it does not match, you are not decoding the Asset this corpus describes and
  every other comparison is meaningless.
- `width` / `height` — the **Canonical Buffer** dimensions, *after* EXIF
  orientation is consumed. For the four orientation entries that transpose
  (5, 6, 7, 8) these are not the dimensions in the file's own header.
- `pixel_hash` — SHA-256 over
  `b"VER1:" + ascii(width) + b":" + ascii(height) + b":sRGB8:" + <row-major RGB8 bytes>`,
  as 64 lowercase hex characters (`PROFILE.md` §3.9).

Digests appear in `manifest.json` and nowhere else — not in this README, not in
`PROFILE.md` — so there is no second copy to drift.

## What each Asset pins

| Asset | Container | Canonical Buffer | What it makes testable |
| --- | --- | --- | --- |
| `jpeg-q92-444.jpg` | JPEG, quality 92, 4:4:4 | 128×96 | IDCT precision with no chroma subsampling |
| `jpeg-q75-420.jpg` | JPEG, quality 75, 4:2:0 | 128×96 | **Chroma upsampling** — the disagreement E12 names first |
| `jpeg-orientation-1.jpg` … `-4.jpg` | JPEG, EXIF 274 = 1…4 | 96×64 | The orientation map, upright and mirrored/rotated in-plane |
| `jpeg-orientation-5.jpg` … `-8.jpg` | JPEG, EXIF 274 = 5…8 | 64×96 | The four orientations that **transpose the dimensions** |
| `jpeg-orientation-out-of-range.jpg` | JPEG, EXIF 274 = 99 | 96×64 | Out-of-range orientation is upright, not an error |
| `jpeg-no-exif.jpg` | JPEG, no EXIF block | 96×64 | Descriptive metadata does not move `pixel_hash` (E1) |
| `png-rgb.png` | PNG, RGB8 | 112×84 | Baseline lossless decode |
| `png-srgb-icc.png` | PNG, RGB8 + embedded sRGB ICC | 112×84 | The ICC → sRGB transform through littleCMS |
| `webp-lossless.webp` | WebP, lossless | 112×84 | WebP lossless decode |
| `tiff-lzw.tiff` | TIFF, LZW | 112×84 | TIFF LZW decode |
| `tiff-uncompressed.tiff` | TIFF, uncompressed | 112×84 | TIFF strip layout, codec-independently |
| `png-rgba-alpha.png` | PNG, RGBA8, diagonal alpha ramp | 96×96 | Alpha composited over white in the gamma domain |
| `gif-palette-transparency.gif` | GIF, palette colour key | 80×64 | Palette transparency is transparency |
| `gif-animated-2frame.gif` | GIF, 2 full-screen frames | 64×48 | Frame 0, and only frame 0 |
| `gif-frame0-subrectangle.gif` | GIF, 2 frames, frame 0 a 40×30 rectangle at (12, 9) on a 64×48 screen | 64×48 | Errata **E7**: coalesce onto the logical screen, transparent canvas, background colour not consumed, disposal never applied |

**The five 112×84 entries share one `pixel_hash`.** One picture in PNG, WebP and
two TIFF codecs, plus the sRGB-tagged PNG, must land on one Canonical Buffer.
That single repeated digest is a statement the corpus can enforce and a
paragraph cannot.

**The eight orientation entries carry eight distinct `pixel_hash` values.** They
are the same picture at the same encoder settings, differing only in EXIF tag
274. A decoder that ignores orientation reproduces at most one of them.

**Two entries publish dimensions their file headers do not.**
`gif-frame0-subrectangle.gif`'s frame rectangle is 40×30 and its expected
Canonical Buffer is 64×48; the eight orientation entries' JPEG headers all say
96×64 and four of them expect 64×96. Both are the point: `width` and `height` in
`manifest.json` are Canonical Buffer dimensions, after CPNP-1 has run.

## Evaluating an implementation

The rule is E12's and it is unforgiving: **reproduce every expected digest
exactly, or you are not conformant for `verd-pillow/1`.** No tolerance, no
partial credit, no per-format subset.

For each entry in `manifest.json`:

1. Read the file at `path` and check `sha256(bytes) == content_sha256`. A
   mismatch means the Asset is not the one this corpus describes — stop.
2. Run CPNP-1 over those bytes to a Canonical Buffer: 8-bit sRGB, `[H, W, 3]`,
   row-major, C-contiguous, per `PROFILE.md` §3.
3. Check the buffer's dimensions against `width` and `height`.
4. Compute SHA-256 over
   `b"VER1:" + ascii(width) + b":" + ascii(height) + b":sRGB8:" + buffer_bytes`
   and check it against `pixel_hash`.

Any single failure means the implementation is **not** conformant for this
profile. It may still produce Records; those Records' identity claims stay
PROVISIONAL in E12's sense (`PROFILE.md` §4).

### In this repository

The conformance run, executed on every commit:

```
PYTHONPATH=services/verd/src .venv/bin/python -m pytest tests/test_record_decoder_corpus.py -q
```

And the corpus's own provenance check — regenerate every image into a temporary
directory and byte-compare against the committed tree, `manifest.json` included:

```
.venv/bin/python scripts/make_decoder_corpus.py --check
```

### For any other implementation

Port this; it needs nothing from this repository except the corpus directory.

```python
import hashlib
import json
import pathlib

RELEASE = pathlib.Path("standards/decoder-corpus/1")
manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))

failures = []
for entry in manifest["files"]:
    data = (RELEASE / entry["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["content_sha256"]:
        failures.append((entry["path"], "not the Asset this corpus describes"))
        continue

    # Your CPNP-1: bytes in; width, height and row-major RGB8 buffer bytes out.
    width, height, pixels = canonicalize(data)

    preimage = b"VER1:%d:%d:sRGB8:" % (width, height) + pixels
    digest = hashlib.sha256(preimage).hexdigest()
    if (width, height) != (entry["width"], entry["height"]) or digest != entry["pixel_hash"]:
        failures.append((entry["path"], f"got {width}x{height} {digest}"))

total = len(manifest["files"])
print(f"{total - len(failures)}/{total} reproduced")
for path, why in failures:
    print("FAIL", path, why)
```

Conformant is `0 failures`. Anything else is not a score.

### Publishing an evaluation

An evaluation is only useful if somebody else can read it. State, outside any
Record: the profile identifier (`verd-pillow/1`), the corpus identifier
(`ver-decoder-corpus/1`), your own toolchain versions in the shape of
`PROFILE.md` §2, the count reproduced out of the total, and — if anything
failed — which paths and what you got. A published evaluation by an
independently built toolchain is what `standards/ver/1.1-draft/RELEASE.md`
row 14 and `VERSIONING.md` §3.4 conditions 1–2 are asking for; this corpus makes
that evaluation possible but does not perform it.

## Rules for changing anything here

1. **This release directory is immutable once published.** Corrections,
   additions and toolchain bumps ship as `standards/decoder-corpus/2/` under a
   new profile identifier. Editing a published digest silently retracts every
   conformance claim made against it.
2. **Every digest is computed, never invented** — by
   `verd.cpnp.canonicalize` + `verd.hashing.pixel_hash` over the committed
   bytes, in `scripts/make_decoder_corpus.py`.
3. **The committed bytes are the corpus.** The generator documents how they were
   made and proves it with `--check`; it does not get to redefine them. If a
   future encoder emits different bytes for the same drawing, `--check` fails,
   and the answer is release 2.
4. **A gap is named, not omitted.** `PROFILE.md` §6 lists what release 1 does
   not pin. Anything added later goes there first.

## The manifest's own digest

`manifest.json` is the corpus's expected-digest ledger, and a ledger needs an
identity of its own: its SHA-256 is

```
d743f077778d60b724f060355a26ea138c9f92ee1d82c275a76707a073e09b5e
```

recorded here rather than inside the file, the same discipline the registry
and the release directories use. `scripts/make_decoder_corpus.py --check`
regenerates the manifest byte-for-byte, so a drifted manifest fails both this
digest and the regeneration check.
