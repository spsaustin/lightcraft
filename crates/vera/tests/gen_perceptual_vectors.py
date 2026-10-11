#!/usr/bin/env python3
"""Generate crates/vera/tests/perceptual-vectors.json with the reference implementations.

The oracles are the ones VER errata E22 names:
  * pdq/1.0      -- Meta's PDQ through `threatexchange` (its `pdq_hasher`, which wraps the C++
                    reference via `pdqhash`), hex in ThreatExchange's own format;
  * phash-dct/64 -- `imagehash.phash` (PIL convert("L"), LANCZOS resize to 32x32, scipy DCT).

Inputs are 8-bit RGB PNGs, so no decoder difference can leak in:
  * the 21 decoder-corpus Canonical Buffers as lightcraft-ver produces them, exported by
      VER_EXPORT_DIR=<dir> cargo test -p lightcraft-ver --test integration export_canonical_pngs -- --ignored
    (each file is named after the corpus path with "/" replaced by "__", plus ".png");
  * the ten reference-set images, read as they are (untagged RGB PNGs: their own Canonical Buffer).

Usage (in a venv with Pillow, numpy, scipy, imagehash, threatexchange):
  python gen_perceptual_vectors.py <export-dir> <crates/ver/standard> > perceptual-vectors.json
"""

import io
import json
import pathlib
import sys

import imagehash
from PIL import Image
from threatexchange.signal_type.pdq.pdq_hasher import pdq_from_bytes


def vector(path_label, png_bytes):
    image = Image.open(io.BytesIO(png_bytes))
    if image.mode != "RGB":
        raise SystemExit(f"{path_label}: expected an RGB PNG, got {image.mode}")
    pdq_hex, quality = pdq_from_bytes(png_bytes)
    # imagehash.phash's own first step, kept so the resample can be checked on its own.
    l32 = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS).tobytes().hex()
    return {
        "path": path_label,
        "width": image.width,
        "height": image.height,
        "pdq": pdq_hex,
        "pdq_quality": quality,
        "phash": str(imagehash.phash(image)),
        "l32": l32,
    }


def synthetic(w, h):
    """A deterministic RGB8 pattern the Rust tests rebuild (`synthetic_rgb` there): a ramp, an
    XOR texture and a quadratic one, so the Jarosz windows and the Lanczos supports are wide."""
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data += bytes(((x * 255) // max(w - 1, 1), ((x ^ y) * 3) & 255, ((x * x + y * 7) >> 3) & 255))
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), bytes(data)).save(buf, "PNG")
    return buf.getvalue()


def main():
    export_dir = pathlib.Path(sys.argv[1])
    standard = pathlib.Path(sys.argv[2])
    out = []
    manifest = json.loads((standard / "decoder-corpus/1/manifest.json").read_text())
    for entry in manifest["files"]:
        label = "decoder-corpus/1/" + entry["path"]
        png = export_dir / (entry["path"].replace("/", "__") + ".png")
        out.append(vector(label, png.read_bytes()))
    for image in sorted((standard / "reference-sets/images").glob("*.png")):
        out.append(vector("reference-sets/images/" + image.name, image.read_bytes()))
    for w, h in ((1200, 900), (3001, 1999), (640, 64), (64, 64), (5, 7)):
        out.append(vector(f"synthetic:{w}x{h}", synthetic(w, h)))
    versions = {
        "imagehash": imagehash.__version__,
        "pillow": Image.__version__,
    }
    try:
        import importlib.metadata as md

        versions["threatexchange"] = md.version("threatexchange")
        versions["pdqhash"] = md.version("pdqhash")
        versions["scipy"] = md.version("scipy")
        versions["numpy"] = md.version("numpy")
    except Exception:  # pragma: no cover - informational only
        pass
    json.dump({"generator": "crates/vera/tests/gen_perceptual_vectors.py", "oracles": versions, "vectors": out}, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
