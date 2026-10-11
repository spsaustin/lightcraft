# VER conformance fixture corpus

Machine-checkable test vectors for the VER record format and the conformance profile
implemented by `packages/ver-validator`. Two artifact releases are covered side by side:
**1.0.1**, judged by `vera-profile/1.0`, and the unratified **1.1 draft**, judged by
`vera-profile/1.1-draft`. Every fixture declares which one it is a statement about — see
[1.1-draft fixtures](#11-draft-fixtures).

The corpus answers one question per file: **given this record, what must a conformant
validator say?** `manifest.json` holds the expected answer for every fixture, so the corpus
is executable rather than illustrative — a validator either reproduces the manifest exactly
or it disagrees with the standard.

| | count | of which 1.1-draft |
| --- | --- | --- |
| conformant records (`valid/`) | 14 | 4 |
| non-conformant / warning records (`invalid/`) | 81 | 29 |
| profile issue codes with a dedicated fixture | 81 of 83 (65 error, 16 warning) | 29 (25 error, 4 warning) |
| codes excused by a documented exemption | 2 (VER001, VER607) | 0 |

Every issue code in the `ver_validator` registry has exactly one fixture registered against
it, or an explicit `coverage_exemptions` entry saying why it cannot have one. A code with
neither, or two fixtures claiming the same code, is a corpus defect. That constraint is
global and has no schema axis: one code, one fixture, whichever release the fixture targets.
Three numbers carry no fixture:

| code | why | covered instead by |
| --- | --- | --- |
| **VER1303** `vector-ref-unverified` | Allocated and deliberately never emitted: the profile validator performs no network I/O, and an unfetchable reference is not evidence of non-conformance. Not in the active registry at all, so it needs no exemption. | `tests/test_profile_validator.py::test_stage14_unfetchable_vector_ref_is_not_a_finding` |
| **VER607** `non-finite-vector-component` | Unreachable from a file. `NaN`, `Infinity` and `-Infinity` are JavaScript literals, not RFC 8259 JSON, so a document containing one is rejected at stage 1 as VER101 and never reaches stage 6 — a fixture carrying them would test VER101 under a VER607 filename. The rule exists for the API entry point, which takes an already-parsed object. | `tests/test_profile_validator.py::test_stage6_non_finite_vector_components_are_ver607` |
| **VER001** `validator-internal-error` | Not file-reachable, and must not be. VER001 is the one code that reports on the *validator* rather than the record: it fires when a rule raises, which is a defect in `ver_validator`. A fixture provoking it would pin a specific validator bug — it would stop reproducing the moment the bug was fixed, and it would be the only file here whose expected result was not a statement about the standard. | `tests/test_profile_validator.py::test_a_rule_that_raises_is_ver001_and_makes_the_run_indeterminate` and its four siblings, which monkeypatch a deliberately-raising rule |

The manifest carries both exemptions and their reasons; stage 0 (VER001's stage) is
consequently absent from the corpus's stage coverage, which
`tests/test_conformance_fixtures.py` derives from the exemption list rather than from a
second, drifting copy of the same judgement.

## Layout

```
conformance/
  manifest.json                 the expected result for every fixture (the contract)
  valid/                        records that MUST validate with no error-severity issue
  invalid/                      records seeded with exactly one defect each
  lineage-peers/
    lineage-cycle-peer.json     a peer record for the --lineage-peer cycle search (VER1604)
  registries/
    conflicting-spaces.json     published space descriptors for the --spaces check (VER402)
    test-registry-snapshot.json corpus-local VER Registry snapshot for --registry (VER1205/6)
  keys/
    corpus-signing-key.pub.pem  public key for the corpus's detached-JWS fixtures
```

`valid/` and `invalid/` hold fixtures; every other directory holds *inputs to a run*. The
manifest-and-disk agreement check covers the two fixture directories only, which is why a
lineage peer — itself a well-formed record — lives in `lineage-peers/` rather than in
`valid/`: it has no expected result of its own, and registering one would claim it does.

## Running the corpus

Against a single fixture, with the validator CLI. No install is needed — from the
repository root:

```
PYTHONPATH=packages/ver-validator/src .venv/bin/python -m ver_validator.cli \
    conformance/valid/l3-full.json
```

`pip install -e packages/ver-validator` provides the `ver-validate` entry point on `PATH`
in an environment whose virtualenv has `pip` (this repository's `.venv` does not). The two
invocations are the same program, and the examples below use the short spelling:

```
ver-validate conformance/valid/l3-full.json
ver-validate --format json conformance/invalid/ver501-undeclared-space.json
```

Exit status is `0` when every named record is conformant, `1` when any carries an
error-severity issue, `2` for a bad invocation. Warnings never change the exit status. Every
report also carries a `status` of `conformant`, `nonconformant` or `indeterminate`; no
fixture in this corpus can produce the third, which is why VER001 is exempt from needing one.

Three of the 1.0.1 fixtures need a flag to reach the rule they test; the manifest entry
names it in each case. Without the flag all three are conformant — the size and registry
fixtures emit nothing at all, and the signature fixture reports VER1403 rather than a failed
verification:

```
ver-validate --max-bytes 1024 conformance/invalid/ver103-record-exceeds-hard-cap.json
ver-validate --spaces conformance/registries/conflicting-spaces.json \
             conformance/invalid/ver402-space-registry-conflict.json
ver-validate --signature-key conformance/keys/corpus-signing-key.pub.pem \
             conformance/invalid/ver1402-signature-verification-failed.json
```

Every 1.1-draft fixture needs `--schema 1.1-draft`, and four of them need a flag beyond it;
the [1.1-draft fixtures](#11-draft-fixtures) section lists those invocations.

Across the whole corpus, from the repository root:

```
pytest tests/test_conformance_fixtures.py -q
```

That suite walks `manifest.json` and asserts the contract below. To run the same checks
without pytest — for instance against a validator that is not this one:

```python
import json
import pathlib

from ver_validator import validate_file
from ver_validator.validator import load_record, load_registry, load_registry_snapshot

root = pathlib.Path("conformance")
manifest = json.loads((root / "manifest.json").read_text())


def run(entry, override):
    spec = {**entry, **override}
    kwargs = {"schema": spec.get("schema", manifest["default_schema"])}
    for key in ("level", "max_bytes", "profile"):
        if spec.get(key):
            kwargs[key] = spec[key]
    if spec.get("spaces_registry"):
        kwargs["spaces_registry"] = load_registry(root / spec["spaces_registry"])
    if spec.get("signature_key"):
        kwargs["signature_key"] = (root / spec["signature_key"]).read_bytes()
    if spec.get("registry"):
        kwargs["registry_snapshot"] = load_registry_snapshot(root / spec["registry"])
    if spec.get("lineage_peers"):
        kwargs["lineage_peers"] = [load_record(root / p) for p in spec["lineage_peers"]]
    return validate_file(root / entry["file"], **kwargs)


for entry in manifest["valid"]:
    for override in [{}, *entry.get("also", [])]:
        report = run(entry, override)
        allowed = set(override.get("expect_codes", entry["allowed_warnings"]))
        assert report.ok and set(report.codes) <= allowed, (entry["file"], report.codes)

for entry in manifest["invalid"]:
    for override in [{}, *entry.get("also", [])]:
        report = run(entry, override)
        spec = {**entry, **override}
        assert set(report.codes) == set(spec["expect_codes"]), (entry["file"], report.codes)
        assert report.ok == (spec["expect_severity"] != "error"), entry["file"]
```

`invalid/ver101-unparseable-json.json` is deliberately not parseable JSON. Any sweep that
parses every `*.json` in the tree must skip it; the manifest flags it in `notes`.

## The manifest contract

```jsonc
{
  "manifest_version": "1.1",     // 1.1 added the registry/lineage_peers/profile options
  "corpus": "ver-conformance/1.1-draft",
  "profile": "vera-profile/1.0", // the profile a run reports when no entry overrides `schema`
  "default_schema": "1.0.1",     // schema release used when an entry names none
  "valid":   [ /* conformant records */ ],
  "invalid": [ /* seeded-defect records */ ]
}
```

`corpus` names the newest release the corpus covers, not the only one: `default_schema`
stays `1.0.1`, so an entry that names no `schema` is still a statement about 1.0.1 and the
existing fixtures are untouched. `manifest_version` went to `1.1` because the three options
below are not decoration — a reader that ignores `lineage_peers` runs the VER1604 fixture
with nothing to walk and gets a different answer from the one the manifest states.

**Two different things are called `profile`, and the collision is worth naming.** The header
field is the *conformance profile identifier* a report carries, and a 1.1-draft run reports
`vera-profile/1.1-draft` regardless of what the header says — the header states the default
for the corpus, which is `default_schema`'s profile. The per-entry `profile` option below is
the *strictness profile* passed to the validator: `default` or `high-trust`. They are
unrelated axes: `high-trust` is a way of reading the 1.1 draft's rules, not a fourth release.

A **`valid` entry**:

| field | meaning |
| --- | --- |
| `file` | path relative to `conformance/` |
| `description` | what the fixture demonstrates |
| `allowed_warnings` | codes that MAY be emitted. Any other code — of any severity — is a failure, and `ok` MUST be true |

An **`invalid` entry**:

| field | meaning |
| --- | --- |
| `file` | path relative to `conformance/` |
| `target_code` | the code this fixture exists to exercise; the filename starts with it |
| `description` | the seeded defect, in the standard's own terms |
| `expect_codes` | the **exact set** of codes the validator must emit — not a subset |
| `expect_severity` | `error` (⇒ `ok` false), `warning` (⇒ `ok` true), or `none` (no issues at all) |

Optional on either kind:

| field | meaning |
| --- | --- |
| `schema` | `1.0.0`, `1.0.1` or `1.1-draft`. `1.1-draft` selects the draft schema **and** the profile rules gated on it |
| `level` | forces the effective conformance level |
| `max_bytes` | the `--max-bytes` hard cap |
| `spaces_registry` | path to a **space** registry of embedding space descriptors (`--spaces`, VER402) |
| `signature_key` | path to a PEM public key (`--signature-key`, stage 15) |
| `registry` | path to a **VER Registry** snapshot of governed value-space tokens (`--registry`, VER1205/VER1206) |
| `lineage_peers` | list of paths to peer records the cycle search may walk (`--lineage-peer`, VER1604) |
| `profile` | `default` or `high-trust` (`--profile`); `high-trust` promotes VER1606 to an error |
| `notes` | operator-facing caveats: required flags, unavoidable collateral codes, how a digest was constructed |

Every path is relative to `conformance/`. `spaces_registry` and `registry` are deliberately
two keys read by two loaders: they name different documents, and one key that accepted
either file would find nothing in the wrong one and report it as a clean run.

`also` holds additional runs of the same file under different options. Each entry is
merged over its parent, so it restates only what changes; `null` clears an inherited
setting (`"signature_key": null` runs the fixture with no key). A valid entry's `also` may
carry `expect_codes`, which then replaces `allowed_warnings` as the permitted set.

The corpus uses `also` for the claims worth pinning down rather than merely asserting in
prose: that the two perceptual-digest fixtures emit their profile code *alone* under
`--schema 1.0.0`, which is precisely the gap the 1.0.1 schema correction closes; that the
signature fixtures behave differently with and without a key and never report an unverified
signature as valid; that the three flag-gated fixtures are conformant when their flag is
absent, so the flag is what reaches the rule; and that `ver1103-c2pa-state-inconsistent`,
audited at `--level L1`, is **silent** — the other half of annex E28's ruling, which tells a
producer that could not validate a manifest to omit `provenance.c2pa` and not claim L3. A
validator that rejected that record below L3 would punish the behaviour the annex demands.

The 1.1-draft fixtures add four more such claims: that a registry snapshot turns VER1206
into VER1205 or into silence depending on what it lists, and never the other way round; that
a lineage cycle is reported only when the peer that closes it was supplied, and its absence
is never read as proof there is none; and that `--profile high-trust` promotes exactly one
code and changes nothing else about the record.

## Fixture construction rules

These rules are what make a fixture a test vector rather than a sample. New fixtures MUST
follow them.

1. **One defect per fixture.** Every fixture is a complete, internally self-consistent
   record except for the single thing it is testing. A fixture that is wrong in two ways
   cannot tell you which rule fired.
2. **`expect_codes` is an exact set, not a floor.** Where a defect provably drags another
   code with it, both are listed and `notes` explains the coupling. The coupled cases in
   this corpus:

   | fixture | also emits | why it cannot be isolated |
   | --- | --- | --- |
   | `ver403-embedding-space-id-malformed` | VER501 | a malformed space_id can never match a declared descriptor, because `spaces[]` entries are pattern-constrained by the schema |
   | `ver705-no-canonical-visual-embedding` | VER201 | VER705 is the profile's defense-in-depth restatement of the schema's `contains` constraint |
   | `ver801-pdq-length`, `ver802-phash-length` | VER201 | the 1.0.1 schema binds digest width to the algorithm; under `--schema 1.0.0` the profile code stands alone |
   | `ver1101-c2pa-raw-present-status-missing` | VER1103 | one record exhibits both readings of the same C2PA inconsistency; the codes overlap by design |
   | `ver1103-c2pa-state-inconsistent` | VER307, VER308, VER1101 | VER1103's *absence* arm is scoped to L3 by annex E28, so the fixture must claim L3 — which drags in every other L3 gate the missing `provenance.c2pa` block fails |
   | `ver306-provenance-missing`, `ver307-c2pa-status-missing` | VER307, VER308 | an absent provenance block fails every L3 gate at once |

   The 1.1-draft fixtures add one coupling with one cause. VER409, VER608, VER609, VER903,
   VER1006, VER1007, VER1207 and VER1605 each also emit **VER201**, because each is a
   profile rule the 1.1 schema *also* encodes — the VER705 precedent, applied eight more
   times. The profile restates them because it has to keep answering when the schema in play
   is relaxed, future or somebody else's, and because two of them (a bundle path handed to a
   filesystem, and which bytes are a vector) are questions a consumer cannot afford two
   answers to. `ver409-bundle-path-illegal`, `ver1006` and `ver1007` emit **two** VER201
   issues rather than one; the set of codes is what the manifest states, and `notes` says so
   in each case.

3. **Every digest, byte string and length is computed, never invented.** Anything a reader
   can recompute from the record MUST be correct: `metadata.raw[].sha256` over exactly the
   carried bytes, `vector_b64` decoding to `dim × dtype_size` bytes (`ceil(dim/8)` for
   `binary`), `vector_ref.byte_length`, `image.source_bytes_ref.sha256` equal to
   `identity.content_hash.value`, and real signatures over the real RFC 8785 JCS payload.

   ```
   # recompute a raw segment digest
   .venv/bin/python -c "import base64,hashlib,json,sys; s=json.load(open(sys.argv[1]))['metadata']['raw'][0]; print(hashlib.sha256(base64.b64decode(s['bytes'])).hexdigest(), s['sha256'])" conformance/valid/l1-raw-metadata.json
   ```

4. **Values that cannot be recomputed are still real digests of a documented preimage.**
   No fixture contains a hand-typed hex string. Where the true input is not carried by the
   record — model weights, a referenced vector blob, the asset bytes themselves — the value
   is `sha256("ver-conformance-corpus/1.0.1 <purpose>")`, e.g.
   `sha256("ver-conformance-corpus/1.0.1 source-bytes boardwalk-sunset.jpg")` for
   `identity.content_hash`. These stand in for values a real producer measures; per errata
   annex E26 they are illustrative and MUST NOT be cited as interoperability evidence.
   The perceptual digests are the leading 64 and 16 hex characters of such a digest: well
   formed, and deliberately not genuine PDQ/pHash outputs — the corpus tests structure, not
   perceptual similarity. The contextual embedding's `source.text_sha256` is the true
   sha256 of `Sunset over a boardwalk, silhouetted pier in background.`, and the redaction
   proofs are `sha256("ver-conformance-corpus/1.0.1 redaction-salt|<field>|<value>")`.
   Documented preimages are dictionary-invertible **by construction** — every
   `salted_sha256`/commitment in this corpus can be reversed from this README,
   which is acceptable for illustrative values (E26) and fatal for real ones. A
   real deployment's salt is never written anywhere a reader can reach (draft
   §5.3, §5.5); 1.1-draft fixtures use the `ver-conformance-corpus/1.1-draft`
   prefix.

   **1.1-draft fixtures use their own prefix**, `sha256("ver-conformance-corpus/1.1-draft
   <purpose>")`, so a value can never be mistaken for one of the 1.0.1 corpus's. The purposes
   in use are `source-bytes <asset>`, `canonical-buffer <asset>`, `weights <checkpoint>`,
   `refset-manifest <set>`, `tokenizer <name>` and `icc-segment <asset>`. Four constructions
   there are not stand-ins at all and recompute from the record itself: `manifest_sha256`,
   `descriptor_sha256`, `recipe_sha256` and `commitment_sha256` — see
   [1.1-draft fixtures](#11-draft-fixtures).
5. **Metadata segment bytes are real container bytes.** The EXIF segment is an APP1 payload
   (`Exif\0\0` + TIFF header + an Orientation entry), the XMP segment carries its
   `http://ns.adobe.com/xap/1.0/\0` identifier prefix, and the IPTC segment is a Photoshop
   IRB holding an 8BIM IPTC-NAA resource — the byte ranges errata annex E11 defines.

   `valid/l3-full.json` is the strictest case, because it is the published 1.0.1 example:
   its `icc` entry is a **complete** JPEG `APP2` ICC payload — `ICC_PROFILE\0`, the two
   sequence octets `01 01`, and a real 588-octet sRGB profile — because an elision is not
   byte-exact preservation under E11 and E26 forbids shipping one at L1 or above. Elsewhere
   in the corpus the ICC and C2PA payloads are short illustrative fragments rather than a
   complete profile or a valid JUMBF manifest store; their digests are still the real
   digests of the bytes shown, and no fixture that carries one declares a level at which
   E26 would forbid it.
6. **Declare full space descriptors** unless the fixture is about not doing so. Annex E13
   enumerates seven `preprocessing` members for a `visual` space (`resize`,
   `interpolation`, `antialias`, `crop`, `mean`, `std`, `channel_order`), five for a `text`
   space (`tokenizer`, `tokenizer_sha256`, `max_length`, `truncation`, `casing`) and none
   for a `fused` space, whose geometry is fixed by its `recipe` (E20). Every space in every
   `valid/` fixture carries its full set, or VER405 would fire and the fixture would no
   longer be silent.
7. **Name the file after the code**: `<code-lowercase>-<slug>.json`, matching
   `target_code`.
8. **Keep fixtures minimal and level-appropriate.** Claim the lowest conformance level that
   still reaches the rule: an L3 claim drags in every L1 and L2 gate, and an unsigned L3
   record always warns VER308. Signature carriage is not level-gated, so the stage-15
   fixtures stay at L1. Where a rule is itself level-scoped — VER1103's absence arm is L3
   only — the fixture must claim the level the rule needs and list the collateral.
9. **Re-run the corpus after any edit** (see above). Editing a signed fixture invalidates
   its signature: re-sign it with the recipe below, or the VER1402/VER1403 expectations
   become meaningless. `valid/l3-full.json` tracks the published 1.0.1 example, so an edit
   to that example means regenerating the fixture from it and re-signing;
   `valid/l3-full-1-1.json` tracks `standards/ver/1.1-draft/example-record-lineage.json` the
   same way and needs no signature, because the published 1.1 example carries none.

### The corpus signing key

`keys/corpus-signing-key.pub.pem` is an Ed25519 public key whose private half is derived
from the published seed `sha256("ver-conformance-corpus/1.0.1 signing-key-seed")`. It
exists so the signature fixtures can be regenerated by anyone, and it carries no trust
whatsoever: it MUST NOT appear in any trust store, and a record signed by it MUST NOT be
treated as attributable.

Re-signing a fixture — Ed25519 is deterministic, so this reproduces the shipped value
byte for byte when the record is unchanged:

```python
import base64, copy, hashlib, json, pathlib
import rfc8785
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

path = pathlib.Path("conformance/valid/l3-full.json")
record = json.loads(path.read_text())

key = Ed25519PrivateKey.from_private_bytes(
    hashlib.sha256(b"ver-conformance-corpus/1.0.1 signing-key-seed").digest()
)
b64u = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

payload = copy.deepcopy(record)             # the payload is the record …
payload["provenance"].pop("signature", None)  # … with the signature removed (annex E17)
header = b64u(json.dumps({"alg": "EdDSA", "kid": "ver-conformance-corpus-2026-08"},
                         separators=(",", ":")).encode())
value = f"{header}..{b64u(key.sign(f'{header}.{b64u(rfc8785.dumps(payload))}'.encode()))}"
record["provenance"]["signature"]["value"] = value
path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
```

`valid/l3-full.json` is the published 1.0.1 example record plus a detached JWS over the
RFC 8785 JCS form of that record with `provenance.signature` removed (errata annex E17).
With no key supplied the profile reports VER1403 — signature present, not verified — which
is why the entry allows that one warning; with the corpus key it verifies and the record
validates with zero issues.

## 1.1-draft fixtures

**Unratified.** Everything in this section describes `standards/ver/1.1-draft/`, which is a
draft: its numbers, its codes and its schema may change before ratification, and no
interoperability claim may rest on a fixture here. The 1.0.1 fixtures are unaffected by all
of it — a 1.1 rule fires only when the run selected `1.1-draft`, and the two sets share a
manifest without sharing a single expectation.

### The schema axis

`"schema": "1.1-draft"` on a manifest entry does two things at once: it validates the record
against `standards/ver/1.1-draft/ver-record.schema.json`, and it switches on the twenty-six
profile rules gated on that release. Those two halves are why a 1.1 fixture cannot be
expressed as an `also` run on a 1.0.1 one — the record bodies differ (`ver_version` `"1.1"`,
members 1.0.x has no place for), not just the options.

Two consequences are worth stating because their opposites look reasonable:

- **`VER104` never fires on a 1.1-draft run**, and `VER105`/`VER106` never fire on a 1.0.x
  one. Section 7 of the draft makes the same three payload quantities normative that 1.0
  leaves advisory, and a severity that changed with the selected release is not something one
  code can express. Exactly one of `{VER104}` and `{VER105, VER106}` is reachable per run.
- **`VER1002` never fires on a 1.1-draft run either.** `salted_sha256` does not exist at
  1.1 — ADR-0006 renamed and reshaped it — so no record could be VER1002-clean and
  VER201-clean at once. Its 1.1 successors are VER1006 and VER1007, and both have fixtures.

### What is here

| fixture | shows |
| --- | --- |
| `valid/lineage-exported.json` | draft §3.5 fixture 1 — same `pixel_hash`, different `content_hash`; accepted |
| `valid/lineage-composed.json` | draft §3.5 fixture 2 — a `compose` event, two `composed` parents, a real `recipe_sha256` |
| `valid/lineage-dangling.json` | draft §3.5 fixture 8 — an unresolvable `record_uri`; **accepted**, and the acceptance is the point |
| `valid/l3-full-1-1.json` | the 1.1 golden: the published draft example, byte for byte |
| `invalid/ver105-…` … `invalid/ver1606-…` | one fixture per new code, 26 in all |
| `lineage-peers/lineage-cycle-peer.json` | the second record of the VER1604 cycle |
| `registries/test-registry-snapshot.json` | the corpus-local VER Registry snapshot |

Draft §3.5 requires eight lineage fixtures before §3 can ratify. Three of them are
`valid/` records (1, 2, 8); the other five are the `invalid/` fixtures for VER1601 through
VER1605, in the order §3.5 lists them.

### Running them

```
ver-validate --schema 1.1-draft conformance/valid/l3-full-1-1.json

ver-validate --schema 1.1-draft --registry conformance/registries/test-registry-snapshot.json \
             conformance/invalid/ver1205-bare-action-token-unregistered.json

ver-validate --schema 1.1-draft \
             --lineage-peer conformance/lineage-peers/lineage-cycle-peer.json \
             conformance/invalid/ver1604-lineage-cycle.json

ver-validate --schema 1.1-draft --profile high-trust \
             conformance/invalid/ver1606-lineage-parent-content-hash-absent.json
```

Each of the four flag-gated fixtures carries an `also` run without its flag, so the corpus
states what the flag *is*: `--registry` turns VER1206 into VER1205 or into silence,
`--lineage-peer` is the only thing that can make a cycle visible, and `--profile high-trust`
turns the VER1606 warning into an error and does nothing else.

### The lineage-peers directory

`lineage-peers/` holds records supplied to a run with `--lineage-peer`. They are inputs, not
fixtures: they have no `target_code`, no expected result, and no manifest entry of their
own, which is why they are not in `valid/` — the manifest-and-disk agreement check covers
`valid/` and `invalid/` only, and a peer registered as a fixture would be claiming a verdict
nobody computed.

A peer matters because VER1604 walks a graph and this validator fetches nothing. A record
is identified in that graph by its `identity.content_hash.value` (the content-hash edge) and
by any `uri` on a chain event whose `target` is `record` (the record-URI edge). `pixel_hash`
is deliberately not an identifier: annex E27 makes it a rendered-identity *grouping* key
that resolves to a candidate and never to an identity, so matching two records by it would
merge unrelated assets into one node and invent cycles that do not exist. The cycle fixture
uses one edge of each type, so a single run exercises the detection rule on both.

### The corpus-local registry snapshot

`registries/test-registry-snapshot.json` is a **corpus-local stand-in** for a VER Registry
snapshot. It is not `standards/registry/snapshots/1/registry.json` and it must never be
substituted for it: it carries **zero trust**, exactly as `keys/corpus-signing-key.pub.pem`
does, and nothing may cite it as evidence that any token is registered anywhere.

It exists for one reason. The corpus's expected codes must not depend on a governed artifact
that changes on its own schedule. A fixture pinning "this token is active" against the
published registry would break the day the registry deprecated that token — and deprecation
never invalidates history, so the fixture would be wrong about the standard rather than the
other way round. The rule under test only asks whether a snapshot lists a token as `active`,
so any spelling works; this one lists `watermark` and nothing else, which makes `watermark`
resolvable and `sidecar_merge` a token that was never registered.

`provenance.registry.sha256` in the two fixtures that name it is the real SHA-256 over the
RFC 8785 JCS serialization of that snapshot document, and moves if the snapshot is edited.

### Digests, and how each was constructed

Rules 3 and 4 apply unchanged. Stand-in values use the 1.1 preimage
`sha256("ver-conformance-corpus/1.1-draft <purpose>")`. Four values are not stand-ins at all
and recompute from what the record itself carries — three of them in one line each, and
`commitment_sha256` below:

```python
import hashlib, rfc8785
jcs = lambda value: hashlib.sha256(rfc8785.dumps(value)).hexdigest()

manifest_sha256   = jcs({"files": files})                                  # §4.2
descriptor_sha256 = jcs({k: v for k, v in space.items() if k != "descriptor_sha256"})
recipe_sha256     = jcs(envelope)                                          # §3.4
```

`valid/lineage-composed.json`'s envelope is exactly:

```json
{"recipe_version":"1","canonicalization":"rfc8785-jcs",
 "steps":[{"op":"canvas","width":8,"height":6,"background":"#000000"},
          {"op":"place","source":"parent-a","x":0,"y":0},
          {"op":"place","source":"parent-b","x":4,"y":0}]}
```

and it validates against `standards/ver/1.1-draft/ver-recipe.schema.json`. The same digest
is the `compose` event's `params_sha256` and the last path segment of `recipe_uri`, so the
three are cross-checkable.

A bundle's per-file `sha256` and `byte_length` recompute too: each file's bytes are the
ASCII string `ver-conformance-corpus/1.1-draft bundle-file <path>`, and each
`model.artifacts[]` entry's are `ver-conformance-corpus/1.1-draft artifact <name>`. The one
exception is `valid/l3-full-1-1.json`, which is the published example and carries the
example's own constructions — `standards/ver/1.1-draft/RELEASE.md` documents every one of
them, including its salt. Nothing in the corpus is a plausible-looking hex string.

`commitment_sha256` is the §5.3 construction, computed:

```
commitment_sha256 := SHA-256( "VER-REDACTION-COMMITMENT-1"
                            ‖ uint64be(len(salt))          ‖ salt
                            ‖ uint64be(len(JCS(value)))    ‖ JCS(value) )
```

with `JCS(value)` the RFC 8785 form of the value **as it appeared in the normalized view**,
so a string is hashed with its quotes. The corpus salt is
`sha256("ver-conformance-corpus/1.1-draft redaction-salt <field>")` — 32 octets, above the
16-octet floor. **That recipe is published here and only here.** The salt appears in no
record, in no `extensions` block, and in no fixture: its absence from the record is the
entire mechanism, and a real deployment's salt is never written down anywhere a reader of
the record can reach. A commitment binds the Producer to a value no later than signing time;
it is not evidence that the value ever existed in the asset.

### `invalid/ver105` is 1.5 MB, deliberately

The cheapest way to trip a single §7 quantity is quantity (c), a single decoded inline
vector above 6 MiB. An inline JSON array is charged at the documented proxy of 8 bytes per
component regardless of dtype, so the limit is reached at 786,433 components — about 1.5 MB
of text when the components are zeros. Quantity (b) would need roughly 8 MB of base64, and
quantity (a) a 10 MiB document. The array is written on one line, as `invalid/ver104` is.

## Relationship to the published artifacts

Fixtures target two artifact releases. The 1.0.x fixtures target release **1.0.1**
(`standards/ver/1.0.1/ver-record.schema.json`) and carry `ver_version` `"1.0"`; the
1.1-draft fixtures target `standards/ver/1.1-draft/ver-record.schema.json` and carry
`ver_version` `"1.1"`.

Each release's flagship fixture tracks that release's published example, and a change to
either example that this corpus does not follow surfaces as a fixture failure — which is the
intent. `valid/l3-full.json` is byte-for-byte the published 1.0.1 example apart from the
added signature. `valid/l3-full-1-1.json` is byte-for-byte
`standards/ver/1.1-draft/example-record-lineage.json` with nothing added: the published 1.1
example carries no `sign` chain event, and §2.3.3 now forbids a signature without one, so
adding a JWS to demonstrate a rule about signatures would be the exact defect class the 1.1
revision exists to remove. The two comparisons are two separate tests rather than one
parametrised helper, because folding them together would mean weakening the stricter one.
