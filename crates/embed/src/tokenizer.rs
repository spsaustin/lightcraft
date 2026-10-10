//! The SigLIP 2 tokenizer: SentencePiece **BPE** (not unigram) with byte fallback, the Gemma
//! vocabulary of 256,000 pieces, read from `tokenizer.model` (a SentencePiece `ModelProto`).
//!
//! The reference is the checkpoint's own `tokenizer.json` as Hugging Face `tokenizers` runs it
//! (what `AutoProcessor` loads for `google/siglip2-base-patch32-256`); golden encodings from it are
//! in `tests/fixtures/siglip2_tokenizer_golden.json`. [`Tokenizer::encode`] does, in order:
//!
//! 1. **Added tokens** first, on the raw text: every CONTROL, UNKNOWN and USER_DEFINED piece
//!    (`<pad>`, `<eos>`, `<bos>`, `<unk>`, `<mask>`, `<unused0>`…, `\n` runs, `▁▁` runs, tab runs,
//!    HTML tags) is matched leftmost-longest and becomes its id; the text between matches is
//!    encoded on its own, so nothing ever merges across an added token.
//! 2. **Normaliser**: each space (U+0020) becomes `▁` (U+2581). No dummy prefix (the model's
//!    normaliser is `identity` with `add_dummy_prefix = false`), no whitespace collapsing, no
//!    Unicode normalisation. The `Split(" ")` pre-tokeniser of `tokenizer.json` runs after that
//!    replacement and so never splits: each stretch between added tokens is one BPE word.
//! 3. **BPE**: start from the characters; a character that is not a piece becomes its UTF-8 bytes
//!    as `<0xXX>` pieces (byte fallback; those never merge). Then repeatedly merge the adjacent pair
//!    whose concatenation is a piece, best first: the higher piece score, then the longer left
//!    part, then the longer right part (in characters), then the lower id, ties by position. That
//!    is the order of the `merges` transformers' converter writes into `tokenizer.json` (checked
//!    against all 580,604 of them). `split_digits` keeps ASCII digits out of every merge.
//!
//! The checkpoint does **not** lower-case in `tokenizer.json` (the `do_lower_case` of its
//! `tokenizer_config.json` is ignored by the fast tokenizer), but SigLIP 2 was trained on
//! lower-cased text, and transformers' `Siglip2Tokenizer` adds a `Lowercase` normaliser before
//! step 2. [`Tokenizer::encode_lowercase`] is that: added tokens matched on the raw text, then each
//! character between them lower-cased on its own (`char::to_lowercase`, as `tokenizers` does, so
//! no final-sigma rule). [`Tokenizer::encode_padded`], what the text tower takes, uses it, then
//! appends `<eos>` (id 1) and pads with `<pad>` (id 0) to [`crate::CONTEXT`], cutting first.
//!
//! The protobuf reader is hand-written and bounded (like `crates/faces/src/onnx.rs`): it needs
//! only `ModelProto.pieces` (field 1), each `SentencePiece`'s `piece` (1), `score` (2) and `type`
//! (3), `TrainerSpec` (2) for `model_type` (3), `split_digits` (25) and `byte_fallback` (35), and
//! `NormalizerSpec` (3) for `name` (1), `precompiled_charsmap` (2), `add_dummy_prefix` (3) and
//! `remove_extra_whitespaces` (4), to check the file is what this encoder expects (tags as in
//! SentencePiece's `sentencepiece_model.proto`). Everything else is skipped by wire type. The file
//! is capped in size and every length is checked against the bytes that remain.
//!
//! One quirk of the Gemma conversion is kept: the byte piece `<0x09>` is the vocabulary entry for
//! a tab character (`\t`), as in `tokenizer.json`.
//!
//! The added-token matching, merge order and byte fallback follow Hugging Face `tokenizers` and
//! the `transformers` slow-to-fast converter (Apache-2.0), re-implemented here.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::io::Read;
use std::path::Path;

use crate::{Error, Result};

/// Largest `tokenizer.model` read (the real one is 4.2 MB).
pub const MAX_MODEL_BYTES: u64 = 64 << 20;
/// `<pad>`.
pub const PAD: u32 = 0;
/// `<eos>`.
pub const EOS: u32 = 1;

/// Most pieces accepted (the real model has 256,000).
const MAX_PIECES: usize = 1 << 20;
/// Longest piece accepted, in bytes (the longest real one is a run of 31 `▁`, 93 bytes).
const MAX_PIECE_BYTES: usize = 1024;
/// `TrainerSpec.ModelType.BPE`.
const MODEL_BPE: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl Kind {
    fn from_proto(v: u64) -> Option<Kind> {
        Some(match v {
            1 => Kind::Normal,
            2 => Kind::Unknown,
            3 => Kind::Control,
            4 => Kind::UserDefined,
            5 => Kind::Unused,
            6 => Kind::Byte,
            _ => return None,
        })
    }
}

pub struct Tokenizer {
    /// Piece strings by id, spelt as in `tokenizer.json`.
    pieces: Vec<String>,
    scores: Vec<f32>,
    /// Piece → id.
    ids: HashMap<String, u32>,
    /// `<0xXX>` piece ids by byte, when the model uses byte fallback.
    bytes: Option<[u32; 256]>,
    unk: Option<u32>,
    /// Added tokens (control, unknown, user-defined) by first byte, longest first.
    added: HashMap<u8, Vec<u32>>,
    split_digits: bool,
}

// ---------------------------------------------------------------------------------------------
// protobuf

enum Wire<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
    Fixed64,
}

/// A bounded reader of one protobuf message held in `buf`.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn malformed(what: &str) -> Error {
    Error::Tokenizer(format!("tokenizer.model is not a SentencePiece model: {what}"))
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self.buf.get(self.pos).ok_or_else(|| malformed("it ends inside a field"))?;
        self.pos = self.pos.checked_add(1).ok_or_else(|| malformed("offset overflow"))?;
        Ok(b)
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..70u32).step_by(7) {
            let b = self.byte()?;
            if shift == 63 && b > 1 {
                return Err(malformed("a number overflows 64 bits"));
            }
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(malformed("a number is too long"))
    }

    /// The next `n` bytes, which must fit in what remains of the message.
    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let n = usize::try_from(n).map_err(|_| malformed("a field is too long"))?;
        let end = self.pos.checked_add(n).ok_or_else(|| malformed("offset overflow"))?;
        let out = self.buf.get(self.pos..end).ok_or_else(|| malformed("a field runs past its message"))?;
        self.pos = end;
        Ok(out)
    }

    /// The next field, or `None` at the end of the message.
    fn next(&mut self) -> Result<Option<(u32, Wire<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let tag = self.varint()?;
        let number = u32::try_from(tag >> 3).map_err(|_| malformed("field number too large"))?;
        if number == 0 || number > 0x1fff_ffff {
            return Err(malformed("invalid field number"));
        }
        let wire = match tag & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let len = self.varint()?;
                Wire::Bytes(self.take(len)?)
            }
            5 => {
                let arr: [u8; 4] = self.take(4)?.try_into().map_err(|_| malformed("a 32-bit field is short"))?;
                Wire::Fixed32(u32::from_le_bytes(arr))
            }
            _ => return Err(malformed("unsupported field type")),
        };
        Ok(Some((number, wire)))
    }
}

struct RawPiece {
    piece: String,
    score: f32,
    kind: Kind,
}

fn parse_piece(buf: &[u8]) -> Result<RawPiece> {
    let mut r = Reader::new(buf);
    let mut out = RawPiece { piece: String::new(), score: 0.0, kind: Kind::Normal };
    while let Some((n, w)) = r.next()? {
        match (n, w) {
            (1, Wire::Bytes(b)) => {
                if b.len() > MAX_PIECE_BYTES {
                    return Err(malformed("a piece is too long"));
                }
                out.piece = std::str::from_utf8(b).map_err(|_| malformed("a piece is not UTF-8"))?.to_string();
            }
            (2, Wire::Fixed32(bits)) => {
                let score = f32::from_bits(bits);
                if score.is_nan() {
                    return Err(malformed("a piece score is not a number"));
                }
                // + 0.0 turns -0.0 into 0.0, so scores order as plain numbers
                out.score = score + 0.0;
            }
            (3, Wire::Varint(v)) => out.kind = Kind::from_proto(v).ok_or_else(|| malformed("unknown piece type"))?,
            (1..=3, _) => return Err(malformed("a piece field has the wrong type")),
            _ => {}
        }
    }
    if out.piece.is_empty() {
        return Err(malformed("an empty piece"));
    }
    Ok(out)
}

/// What `from_bytes` checks in `TrainerSpec` and `NormalizerSpec` (proto defaults when absent).
struct Spec {
    model_type: u64,
    split_digits: bool,
    byte_fallback: bool,
    normalizer: String,
    charsmap: bool,
    dummy_prefix: bool,
    collapse_whitespace: bool,
}

fn parse_trainer(buf: &[u8], spec: &mut Spec) -> Result<()> {
    let mut r = Reader::new(buf);
    while let Some((n, w)) = r.next()? {
        match (n, w) {
            (3, Wire::Varint(v)) => spec.model_type = v,
            (25, Wire::Varint(v)) => spec.split_digits = v != 0,
            (35, Wire::Varint(v)) => spec.byte_fallback = v != 0,
            _ => {}
        }
    }
    Ok(())
}

fn parse_normalizer(buf: &[u8], spec: &mut Spec) -> Result<()> {
    let mut r = Reader::new(buf);
    while let Some((n, w)) = r.next()? {
        match (n, w) {
            (1, Wire::Bytes(b)) => spec.normalizer = String::from_utf8_lossy(b.get(..b.len().min(64)).unwrap_or_default()).into_owned(),
            (2, Wire::Bytes(b)) => spec.charsmap = !b.is_empty(),
            (3, Wire::Varint(v)) => spec.dummy_prefix = v != 0,
            (4, Wire::Varint(v)) => spec.collapse_whitespace = v != 0,
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// BPE

/// A queued merge: the pair starting at symbol `pos` would become piece `new_id`.
struct Candidate {
    score: f32,
    left_chars: usize,
    right_chars: usize,
    new_id: u32,
    pos: usize,
}

impl Candidate {
    /// Less is better: higher score, longer left part, longer right part, lower id, earlier.
    fn order(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(other.left_chars.cmp(&self.left_chars))
            .then(other.right_chars.cmp(&self.right_chars))
            .then(self.new_id.cmp(&other.new_id))
            .then(self.pos.cmp(&other.pos))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.order(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order(other)
    }
}

struct Symbol {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
    /// Byte-fallback and unknown symbols never merge.
    fixed: bool,
}

fn push_symbol(symbols: &mut Vec<Symbol>, id: u32, fixed: bool) {
    let n = symbols.len();
    if let Some(last) = symbols.last_mut() {
        last.next = Some(n);
    }
    symbols.push(Symbol { id, prev: n.checked_sub(1), next: None, alive: true, fixed });
}

impl Tokenizer {
    /// Read `tokenizer.model`.
    pub fn load(path: &Path) -> Result<Tokenizer> {
        let fail = |e: std::io::Error| Error::Tokenizer(format!("{}: {e}", path.display()));
        let file = std::fs::File::open(path).map_err(fail)?;
        if file.metadata().map_err(fail)?.len() > MAX_MODEL_BYTES {
            return Err(Error::Tokenizer(format!("{} is larger than {MAX_MODEL_BYTES} bytes", path.display())));
        }
        let mut bytes = Vec::new();
        file.take(MAX_MODEL_BYTES.saturating_add(1)).read_to_end(&mut bytes).map_err(fail)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a `ModelProto` already in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
        if bytes.len() as u64 > MAX_MODEL_BYTES {
            return Err(malformed("larger than allowed"));
        }
        let mut spec = Spec {
            model_type: 1,
            split_digits: false,
            byte_fallback: false,
            normalizer: String::new(),
            charsmap: false,
            dummy_prefix: true,
            collapse_whitespace: true,
        };
        let mut raw = Vec::new();
        let (mut saw_trainer, mut saw_normalizer) = (false, false);
        let mut r = Reader::new(bytes);
        while let Some((n, w)) = r.next()? {
            match (n, w) {
                (1, Wire::Bytes(b)) => {
                    if raw.len() >= MAX_PIECES {
                        return Err(malformed("too many pieces"));
                    }
                    raw.push(parse_piece(b)?);
                }
                (2, Wire::Bytes(b)) => {
                    saw_trainer = true;
                    parse_trainer(b, &mut spec)?;
                }
                (3, Wire::Bytes(b)) => {
                    saw_normalizer = true;
                    parse_normalizer(b, &mut spec)?;
                }
                (1..=3, _) => return Err(malformed("a top-level field has the wrong type")),
                _ => {}
            }
        }
        if raw.is_empty() {
            return Err(malformed("no pieces"));
        }
        if !saw_trainer || spec.model_type != MODEL_BPE {
            return Err(Error::Tokenizer(format!("tokenizer.model is not a BPE model (model_type {})", spec.model_type)));
        }
        if !saw_normalizer || spec.normalizer != "identity" || spec.charsmap || spec.dummy_prefix || spec.collapse_whitespace {
            return Err(Error::Tokenizer(format!(
                "tokenizer.model's normaliser is not the identity one this encoder implements (name {:?}, charsmap {}, dummy prefix {}, \
                 whitespace collapsing {})",
                spec.normalizer, spec.charsmap, spec.dummy_prefix, spec.collapse_whitespace
            )));
        }
        Self::build(raw, &spec)
    }

    fn build(raw: Vec<RawPiece>, spec: &Spec) -> Result<Tokenizer> {
        let has_tab = raw.iter().any(|p| p.piece == "\t");
        let mut pieces = Vec::with_capacity(raw.len());
        let mut scores = Vec::with_capacity(raw.len());
        let mut ids = HashMap::with_capacity(raw.len());
        let mut bytes = [None::<u32>; 256];
        let mut unk = None;
        let mut added: HashMap<u8, Vec<u32>> = HashMap::new();
        for (i, p) in raw.into_iter().enumerate() {
            let id = u32::try_from(i).map_err(|_| malformed("too many pieces"))?;
            let mut piece = p.piece;
            match p.kind {
                Kind::Byte => {
                    let b = byte_piece(&piece).ok_or_else(|| malformed("a byte piece is not <0xXX>"))?;
                    let slot = bytes.get_mut(usize::from(b)).ok_or_else(|| malformed("byte out of range"))?;
                    if slot.is_some() {
                        return Err(malformed("a byte piece is repeated"));
                    }
                    *slot = Some(id);
                    // the Gemma conversion spells this byte piece as the tab it stands for
                    if b == b'\t' && !has_tab {
                        piece = "\t".to_string();
                    }
                }
                Kind::Unknown => {
                    unk = unk.or(Some(id));
                    added.entry(first_byte(&piece)).or_default().push(id);
                }
                Kind::Control | Kind::UserDefined => added.entry(first_byte(&piece)).or_default().push(id),
                Kind::Normal | Kind::Unused => {}
            }
            if ids.insert(piece.clone(), id).is_some() {
                return Err(malformed("a piece is repeated"));
            }
            pieces.push(piece);
            scores.push(p.score);
        }
        let bytes = if spec.byte_fallback {
            let mut table = [0u32; 256];
            for (slot, id) in table.iter_mut().zip(bytes.iter()) {
                *slot = id.ok_or_else(|| malformed("byte fallback is on but a byte piece is missing"))?;
            }
            Some(table)
        } else {
            None
        };
        for list in added.values_mut() {
            list.sort_by_key(|id| (Reverse(pieces.get(*id as usize).map_or(0, String::len)), *id));
        }
        Ok(Tokenizer { pieces, scores, ids, bytes, unk, added, split_digits: spec.split_digits })
    }

    /// Pieces in the vocabulary.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The token ids of `text` without padding or `<eos>`, exactly as the checkpoint's
    /// `tokenizer.json` encodes it: no lower-casing (what the golden tests compare).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, false)
    }

    /// [`Tokenizer::encode`] with transformers' `Siglip2Tokenizer` lower-casing: the text between
    /// added tokens is lower-cased character by character before it is encoded.
    pub fn encode_lowercase(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, true)
    }

    /// `encode_lowercase` plus `<eos>`, cut and padded to `CONTEXT`: what the text tower takes
    /// (SigLIP 2 was trained on lower-cased text).
    pub fn encode_padded(&self, text: &str) -> [u32; crate::CONTEXT] {
        let mut out = [PAD; crate::CONTEXT];
        let ids = self.encode_lowercase(text);
        let keep = ids.len().min(crate::CONTEXT.saturating_sub(1));
        for (slot, id) in out.iter_mut().zip(ids.iter().take(keep)) {
            *slot = *id;
        }
        if let Some(slot) = out.get_mut(keep) {
            *slot = EOS;
        }
        out
    }

    fn encode_with(&self, text: &str, lowercase: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut start, mut at) = (0usize, 0usize);
        while let Some(rest) = text.get(at..) {
            let Some(c) = rest.chars().next() else { break };
            if let Some((id, len)) = self.added_at(rest) {
                if let Some(segment) = text.get(start..at) {
                    self.encode_segment(segment, lowercase, &mut out);
                }
                out.push(id);
                at = at.saturating_add(len);
                start = at;
            } else {
                at = at.saturating_add(c.len_utf8());
            }
        }
        if let Some(segment) = text.get(start..) {
            self.encode_segment(segment, lowercase, &mut out);
        }
        out
    }

    /// The longest added token `text` starts with, and its length in bytes.
    fn added_at(&self, text: &str) -> Option<(u32, usize)> {
        let list = self.added.get(text.as_bytes().first()?)?;
        list.iter().find_map(|id| {
            let piece = self.pieces.get(*id as usize)?;
            text.starts_with(piece.as_str()).then_some((*id, piece.len()))
        })
    }

    /// One stretch of text between added tokens: normalise, then BPE.
    fn encode_segment(&self, segment: &str, lowercase: bool, out: &mut Vec<u32>) {
        if segment.is_empty() {
            return;
        }
        let mut symbols: Vec<Symbol> = Vec::with_capacity(segment.len());
        let chars = segment.chars().flat_map(|c| {
            let lower = lowercase.then(|| c.to_lowercase());
            let same = (!lowercase).then_some(c);
            lower.into_iter().flatten().chain(same)
        });
        let mut buf = [0u8; 4];
        for c in chars {
            let c = if c == ' ' { '\u{2581}' } else { c };
            let s: &str = c.encode_utf8(&mut buf);
            if let Some(id) = self.ids.get(s) {
                push_symbol(&mut symbols, *id, false);
            } else if let Some(table) = &self.bytes {
                for b in s.bytes() {
                    if let Some(id) = table.get(usize::from(b)) {
                        push_symbol(&mut symbols, *id, true);
                    }
                }
            } else if let Some(unk) = self.unk {
                // consecutive unknown characters fuse into one `<unk>`
                if !symbols.last().is_some_and(|s| s.fixed && s.id == unk) {
                    push_symbol(&mut symbols, unk, true);
                }
            }
        }

        let mut heap: BinaryHeap<Reverse<Candidate>> = (0..symbols.len()).filter_map(|pos| self.candidate(&symbols, pos)).map(Reverse).collect();
        while let Some(Reverse(top)) = heap.pop() {
            // stale when either side has changed since it was queued
            if self.candidate(&symbols, top.pos).is_none_or(|now| now.new_id != top.new_id) {
                continue;
            }
            let Some(right) = symbols.get(top.pos).and_then(|s| s.next) else { continue };
            let after = symbols.get(right).and_then(|s| s.next);
            if let Some(r) = symbols.get_mut(right) {
                r.alive = false;
            }
            if let Some(n) = after.and_then(|a| symbols.get_mut(a)) {
                n.prev = Some(top.pos);
            }
            let Some(merged) = symbols.get_mut(top.pos) else { continue };
            merged.id = top.new_id;
            merged.next = after;
            let prev = merged.prev;
            if let Some(c) = prev.and_then(|p| self.candidate(&symbols, p)) {
                heap.push(Reverse(c));
            }
            if let Some(c) = self.candidate(&symbols, top.pos) {
                heap.push(Reverse(c));
            }
        }

        let mut at = if symbols.is_empty() { None } else { Some(0usize) };
        while let Some(s) = at.and_then(|i| symbols.get(i)) {
            out.push(s.id);
            at = s.next;
        }
    }

    /// The merge of the live symbol at `pos` with the one after it, if their concatenation is a piece.
    fn candidate(&self, symbols: &[Symbol], pos: usize) -> Option<Candidate> {
        let left = symbols.get(pos).filter(|s| s.alive && !s.fixed)?;
        let right = symbols.get(left.next?).filter(|s| s.alive && !s.fixed)?;
        let l = self.pieces.get(left.id as usize)?;
        let r = self.pieces.get(right.id as usize)?;
        if self.split_digits && (l.bytes().any(|b| b.is_ascii_digit()) || r.bytes().any(|b| b.is_ascii_digit())) {
            return None;
        }
        let mut joined = String::with_capacity(l.len().saturating_add(r.len()));
        joined.push_str(l);
        joined.push_str(r);
        let new_id = *self.ids.get(&joined)?;
        Some(Candidate {
            score: self.scores.get(new_id as usize).copied().unwrap_or(f32::NEG_INFINITY),
            left_chars: l.chars().count(),
            right_chars: r.chars().count(),
            new_id,
            pos,
        })
    }
}

/// `<0xXX>` → the byte.
fn byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn first_byte(piece: &str) -> u8 {
    piece.as_bytes().first().copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- a tiny ModelProto writer -----------------------------------------------------------

    fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn field_varint(out: &mut Vec<u8>, n: u32, v: u64) {
        varint(out, u64::from(n) << 3);
        varint(out, v);
    }

    fn field_bytes(out: &mut Vec<u8>, n: u32, b: &[u8]) {
        varint(out, (u64::from(n) << 3) | 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }

    fn piece(p: &str, score: f32, kind: u64) -> Vec<u8> {
        let mut m = Vec::new();
        field_bytes(&mut m, 1, p.as_bytes());
        varint(&mut m, (2 << 3) | 5);
        m.extend_from_slice(&score.to_le_bytes());
        field_varint(&mut m, 3, kind);
        m
    }

    #[derive(Clone, Copy)]
    struct Synth {
        model_type: u64,
        byte_fallback: bool,
        split_digits: bool,
        normalizer: &'static str,
        dummy_prefix: bool,
    }

    const SYNTH: Synth = Synth { model_type: 2, byte_fallback: true, split_digits: true, normalizer: "identity", dummy_prefix: false };

    /// The synthetic model's normal pieces and scores (the higher, the earlier merged).
    const NORMAL: &[(&str, f32)] = &[
        ("a", -1.0),
        ("b", -2.0),
        ("c", -3.0),
        ("\u{2581}", -4.0),
        ("1", -5.0),
        ("2", -6.0),
        ("x", -7.0),
        ("bc", -10.0),
        ("ab", -11.0),
        ("aa", -12.0),
        ("\u{2581}b", -13.0),
        ("12", -14.0),
        ("xyz", -15.0),
        ("y", -16.0),
        ("z", -17.0),
        ("xy", -18.0),
        ("yz", -19.0),
    ];

    fn synth(s: Synth) -> Vec<u8> {
        let mut out = Vec::new();
        for (p, k) in [("<pad>", 3), ("<eos>", 3), ("<bos>", 3), ("<unk>", 2)] {
            field_bytes(&mut out, 1, &piece(p, 0.0, k));
        }
        for b in 0..=255u8 {
            field_bytes(&mut out, 1, &piece(&format!("<0x{b:02X}>"), 0.0, 6));
        }
        field_bytes(&mut out, 1, &piece("<mask>", 0.0, 4));
        field_bytes(&mut out, 1, &piece("\u{2581}\u{2581}", 0.0, 4));
        for (p, score) in NORMAL {
            field_bytes(&mut out, 1, &piece(p, *score, 1));
        }
        let mut trainer = Vec::new();
        field_varint(&mut trainer, 3, s.model_type);
        field_varint(&mut trainer, 4, 300);
        field_varint(&mut trainer, 25, u64::from(s.split_digits));
        field_varint(&mut trainer, 35, u64::from(s.byte_fallback));
        field_bytes(&mut out, 2, &trainer);
        let mut norm = Vec::new();
        field_bytes(&mut norm, 1, s.normalizer.as_bytes());
        field_varint(&mut norm, 3, u64::from(s.dummy_prefix));
        field_varint(&mut norm, 4, 0);
        field_bytes(&mut out, 3, &norm);
        out
    }

    fn tiny() -> Tokenizer {
        Tokenizer::from_bytes(&synth(SYNTH)).unwrap()
    }

    fn id(t: &Tokenizer, p: &str) -> u32 {
        *t.ids.get(p).unwrap_or_else(|| panic!("no piece {p:?}"))
    }

    fn ids(t: &Tokenizer, ps: &[&str]) -> Vec<u32> {
        ps.iter().map(|p| id(t, p)).collect()
    }

    /// The id of byte piece `b` in the synthetic model (after the four control pieces).
    fn byte(b: u8) -> u32 {
        4 + u32::from(b)
    }

    #[test]
    fn synthetic_model_parses() {
        let t = tiny();
        assert_eq!(t.len(), 4 + 256 + 2 + NORMAL.len());
        assert!(!t.is_empty());
        assert_eq!(id(&t, "<pad>"), PAD);
        assert_eq!(id(&t, "<eos>"), EOS);
        assert_eq!(t.unk, Some(3));
        // the byte piece for a tab is spelt as a tab, as in tokenizer.json
        assert_eq!(id(&t, "\t"), byte(9));
        assert!(!t.ids.contains_key("<0x09>"));
    }

    #[test]
    fn merges_highest_score_first() {
        let t = tiny();
        // bc (-10) beats ab (-11); there is no "abc", so a + bc stays
        assert_eq!(t.encode("abc"), ids(&t, &["a", "bc"]));
        assert_eq!(t.encode("ab"), ids(&t, &["ab"]));
        // equal candidates: the leftmost first
        assert_eq!(t.encode("aaa"), ids(&t, &["aa", "a"]));
        assert_eq!(t.encode("aaaa"), ids(&t, &["aa", "aa"]));
        // xy (-18) before yz (-19), then the merged symbol merges on into xyz
        assert_eq!(t.encode("xyz"), ids(&t, &["xyz"]));
    }

    #[test]
    fn spaces_become_the_word_marker_without_a_prefix() {
        let t = tiny();
        assert_eq!(t.encode("a b"), ids(&t, &["a", "\u{2581}b"]));
        assert_eq!(t.encode(" a"), ids(&t, &["\u{2581}", "a"]));
        assert_eq!(t.encode("a "), ids(&t, &["a", "\u{2581}"]));
        // two spaces merge into the user-defined ▁▁ (score 0 beats every normal piece)
        assert_eq!(t.encode("a  b"), ids(&t, &["a", "\u{2581}\u{2581}", "b"]));
        assert_eq!(t.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn byte_fallback_for_unknown_characters() {
        let t = tiny();
        assert_eq!(t.encode("é"), vec![byte(0xC3), byte(0xA9)]);
        assert_eq!(t.encode("aéb"), vec![id(&t, "a"), byte(0xC3), byte(0xA9), id(&t, "b")]);
        assert_eq!(t.encode("🌊"), vec![byte(0xF0), byte(0x9F), byte(0x8C), byte(0x8A)]);
    }

    #[test]
    fn unknown_characters_fuse_without_byte_fallback() {
        let t = Tokenizer::from_bytes(&synth(Synth { byte_fallback: false, ..SYNTH })).unwrap();
        assert_eq!(t.encode("aééb"), vec![id(&t, "a"), 3, id(&t, "b")]);
    }

    #[test]
    fn digits_split_when_the_model_says_so() {
        let t = tiny();
        assert_eq!(t.encode("12"), ids(&t, &["1", "2"]));
        let joined = Tokenizer::from_bytes(&synth(Synth { split_digits: false, ..SYNTH })).unwrap();
        assert_eq!(joined.encode("12"), ids(&joined, &["12"]));
    }

    #[test]
    fn added_tokens_are_matched_first_and_never_merged_across() {
        let t = tiny();
        assert_eq!(t.encode("a<mask>b"), ids(&t, &["a", "<mask>", "b"]));
        assert_eq!(t.encode("a<mask>bc<eos>"), ids(&t, &["a", "<mask>", "bc", "<eos>"]));
        // "ab" would merge, but not across a token
        assert_eq!(t.encode("a<pad>b"), ids(&t, &["a", "<pad>", "b"]));
        // a user-defined piece written out in the text
        assert_eq!(t.encode("a\u{2581}\u{2581}\u{2581}b"), ids(&t, &["a", "\u{2581}\u{2581}", "\u{2581}b"]));
    }

    #[test]
    fn lowercase_runs_between_added_tokens() {
        let t = tiny();
        assert_eq!(t.encode_lowercase("AB C"), t.encode("ab c"));
        assert_ne!(t.encode("AB"), t.encode("ab"));
        // added tokens are matched on the raw text, so "<MASK>" is not "<mask>"
        assert_ne!(t.encode_lowercase("<MASK>"), ids(&t, &["<mask>"]));
        assert_eq!(t.encode_lowercase("<mask>A"), ids(&t, &["<mask>", "a"]));
    }

    #[test]
    fn padded_layout_is_ids_eos_then_pad() {
        let t = tiny();
        let p = t.encode_padded("Ab c");
        let want = t.encode("ab c");
        assert_eq!(&p[..want.len()], want.as_slice());
        assert_eq!(p[want.len()], EOS);
        assert!(p[want.len() + 1..].iter().all(|&x| x == PAD));

        let empty = t.encode_padded("");
        assert_eq!(empty[0], EOS);
        assert!(empty[1..].iter().all(|&x| x == PAD));

        // 100 single-digit tokens: cut to 63, then <eos>
        let p = t.encode_padded(&"1".repeat(100));
        assert!(p[..crate::CONTEXT - 1].iter().all(|&x| x == id(&t, "1")));
        assert_eq!(p[crate::CONTEXT - 1], EOS);
        // exactly 63 tokens fit with their <eos>
        let p = t.encode_padded(&"1".repeat(63));
        assert_eq!(p[62], id(&t, "1"));
        assert_eq!(p[63], EOS);
    }

    #[test]
    fn rejects_models_this_encoder_does_not_implement() {
        assert!(Tokenizer::from_bytes(&synth(Synth { model_type: 1, ..SYNTH })).is_err());
        assert!(Tokenizer::from_bytes(&synth(Synth { normalizer: "nmt_nfkc", ..SYNTH })).is_err());
        assert!(Tokenizer::from_bytes(&synth(Synth { dummy_prefix: true, ..SYNTH })).is_err());
        assert!(Tokenizer::from_bytes(&[]).is_err());
        // byte fallback without the byte pieces
        let mut no_bytes = Vec::new();
        field_bytes(&mut no_bytes, 1, &piece("a", -1.0, 1));
        let mut trainer = Vec::new();
        field_varint(&mut trainer, 3, 2);
        field_varint(&mut trainer, 35, 1);
        field_bytes(&mut no_bytes, 2, &trainer);
        let mut norm = Vec::new();
        field_bytes(&mut norm, 1, b"identity");
        field_varint(&mut norm, 3, 0);
        field_varint(&mut norm, 4, 0);
        field_bytes(&mut no_bytes, 3, &norm);
        assert!(Tokenizer::from_bytes(&no_bytes).is_err());
    }

    // --- hostile input ----------------------------------------------------------------------
    // (proptest is not a dependency of this crate; a seeded xorshift drives the same checks)

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    #[test]
    fn from_bytes_never_panics_on_garbage() {
        let good = synth(SYNTH);
        for n in 0..good.len() {
            let _ = Tokenizer::from_bytes(&good[..n]);
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..2000 {
            let len = rng.below(200);
            let junk: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = Tokenizer::from_bytes(&junk);
        }
        // byte flips in a valid model; whatever still parses must still encode
        for _ in 0..2000 {
            let mut m = good.clone();
            for _ in 0..1 + rng.below(4) {
                let i = rng.below(m.len());
                m[i] = rng.next() as u8;
            }
            if let Ok(t) = Tokenizer::from_bytes(&m) {
                let _ = t.encode_padded("ab c 12 é<mask>  x");
            }
        }
        // huge and overflowing length fields, at the top level and nested
        for len in [u64::MAX, u64::MAX >> 1, 1 << 40, MAX_MODEL_BYTES, 1000] {
            let mut m = Vec::new();
            varint(&mut m, (1 << 3) | 2);
            varint(&mut m, len);
            m.extend_from_slice(b"abc");
            assert!(Tokenizer::from_bytes(&m).is_err(), "length {len}");
            let mut nested = Vec::new();
            field_bytes(&mut nested, 1, &m);
            assert!(Tokenizer::from_bytes(&nested).is_err(), "nested length {len}");
        }
        // an overlong varint, a group, field number 0
        assert!(Tokenizer::from_bytes(&[0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]).is_err());
        assert!(Tokenizer::from_bytes(&[0x0b, 0x0c]).is_err());
        assert!(Tokenizer::from_bytes(&[0x00, 0x00]).is_err());
    }

    fn random_text(rng: &mut Rng) -> String {
        const POOL: &[char] = &[
            'a',
            'b',
            'c',
            'x',
            'y',
            'z',
            'A',
            'B',
            ' ',
            ' ',
            '\u{2581}',
            '1',
            '2',
            '9',
            'é',
            'İ',
            'Σ',
            '東',
            '🌊',
            '\n',
            '\t',
            '<',
            '>',
            '\0',
            '\u{200b}',
            '\u{0301}',
            '\u{10ffff}',
            'ß',
        ];
        let len = rng.below(40);
        let mut s = String::new();
        for _ in 0..len {
            match rng.below(10) {
                0 => s.push_str("<mask>"),
                1 => s.push_str("<eos>"),
                2 => s.push(char::from_u32(rng.next() as u32 % 0x11_0000).unwrap_or('?')),
                _ => s.push(POOL[rng.below(POOL.len())]),
            }
        }
        s
    }

    #[test]
    fn encode_never_panics() {
        let t = tiny();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..5000 {
            let s = random_text(&mut rng);
            assert!(t.encode(&s).iter().all(|&i| (i as usize) < t.len()));
            let _ = t.encode_lowercase(&s);
            assert!(t.encode_padded(&s).contains(&EOS));
        }
    }

    // --- golden encodings from Hugging Face tokenizers -------------------------------------

    #[derive(serde::Deserialize)]
    struct Row {
        text: String,
        ids: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    struct Golden {
        encodings: Vec<Row>,
        lower_encodings: Vec<Row>,
        #[serde(default)]
        special_encodings: Vec<Row>,
    }

    const GOLDEN: &str = include_str!("../tests/fixtures/siglip2_tokenizer_golden.json");

    fn real_model() -> Option<Tokenizer> {
        let Some(dir) = std::env::var_os("LIGHTCRAFT_SIGLIP2_DIR") else {
            println!("note: LIGHTCRAFT_SIGLIP2_DIR is unset; skipping the SigLIP 2 golden tokenizer test");
            return None;
        };
        let path = Path::new(&dir).join(crate::TOKENIZER_FILE);
        if !path.is_file() {
            println!("note: {} is missing; skipping the SigLIP 2 golden tokenizer test", path.display());
            return None;
        }
        Some(Tokenizer::load(&path).unwrap())
    }

    fn check(name: &str, rows: &[Row], f: impl Fn(&str) -> Vec<u32>) -> usize {
        let mut failed = 0;
        for row in rows {
            let got = f(&row.text);
            if got != row.ids {
                failed += 1;
                println!("{name} mismatch for {:?}:\n  want {:?}\n  got  {:?}", row.text, row.ids, got);
            }
        }
        println!("{name}: {}/{} rows match", rows.len() - failed, rows.len());
        failed
    }

    #[test]
    fn golden_matches_hugging_face() {
        let Some(t) = real_model() else { return };
        assert_eq!(t.len(), 256_000);
        let golden: Golden = serde_json::from_str(GOLDEN).unwrap();
        let mut failed = check("encode", &golden.encodings, |s| t.encode(s));
        failed += check("encode_lowercase", &golden.lower_encodings, |s| t.encode_lowercase(s));
        let [padded, plain] = &golden.special_encodings[..] else { panic!("two special rows expected") };
        assert_eq!(t.encode_padded(&padded.text).to_vec(), padded.ids, "padded layout");
        let mut with_eos = t.encode(&plain.text);
        with_eos.push(EOS);
        assert_eq!(with_eos, plain.ids, "<eos> convention");
        // a larger local corpus in the same format (never committed)
        if let Some(extra) = std::env::var_os("LIGHTCRAFT_SIGLIP2_EXTRA_GOLDEN") {
            let extra: Golden = serde_json::from_str(&std::fs::read_to_string(extra).unwrap()).unwrap();
            failed += check("extra encode", &extra.encodings, |s| t.encode(s));
            failed += check("extra encode_lowercase", &extra.lower_encodings, |s| t.encode_lowercase(s));
        }
        assert_eq!(failed, 0);
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for _ in 0..2000 {
            let _ = t.encode_padded(&random_text(&mut rng));
        }
    }
}
