//! Safetensors weights read on demand (positional reads, no memory map, F32 only).
//!
//! Tensors are read from the file when a module asks for them (no memory map: `unsafe` stays in
//! `lightcraft-sysmem`), so loading only the image tower never touches the text tower's bytes, and
//! [`Weights::read_rows`] reads just the rows of the 256,000 × 768 token-embedding table a query
//! needs (64 × 3 KB) instead of its 787 MB.
//!
//! Adapted from `crates/segment/src/weights.rs` (same licence and provenance).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;

use crate::{Error, Result};

/// Largest header accepted (the real one is 49 KB).
const MAX_HEADER: u64 = 100 << 20;
/// Most rows [`Weights::read_rows`] reads in one call.
const MAX_ROWS: usize = 4096;

struct Entry {
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// An open safetensors file: its header, and the file to read tensor bytes from.
pub struct Weights {
    file: Mutex<File>,
    entries: HashMap<String, Entry>,
    data_start: u64,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        let io = |e: std::io::Error| Error::Model(format!("{}: {e}", path.display()));
        let mut file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(io)?;
        let n = u64::from_le_bytes(len);
        if n > MAX_HEADER || n.saturating_add(8) > file_len {
            return Err(Error::Model(format!("{}: not a safetensors file (header of {n} bytes)", path.display())));
        }
        let mut header = vec![0u8; usize::try_from(n).map_err(|_| Error::Model("header too large".into()))?];
        file.read_exact(&mut header).map_err(io)?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|e| Error::Model(format!("{}: bad header: {e}", path.display())))?;
        let mut entries = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match v.get("dtype").and_then(|d| d.as_str()) {
                Some("F32") => DType::F32,
                other => return Err(Error::Model(format!("{name}: unsupported dtype {other:?} (the SigLIP 2 checkpoint is F32)"))),
            };
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().and_then(|x| usize::try_from(x).ok())).collect())
                .unwrap_or_default();
            let offsets: Vec<u64> =
                v.get("data_offsets").and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
            let [start, end] = offsets[..] else { return Err(Error::Model(format!("{name}: bad data_offsets"))) };
            let want = shape.iter().try_fold(dtype.size_in_bytes(), |a, d| a.checked_mul(*d));
            if end.checked_sub(start) != want.map(|w| w as u64) {
                return Err(Error::Model(format!("{name}: size does not match its shape")));
            }
            // the bytes must be in the file (a damaged or truncated download fails here, not later)
            if (8 + n).checked_add(end).is_none_or(|e| e > file_len) {
                return Err(Error::Model(format!("{}: tensor {name} lies outside the file (incomplete or damaged download?)", path.display())));
            }
            entries.insert(name, Entry { shape, start, end });
        }
        Ok(Arc::new(Self { file: Mutex::new(file), entries, data_start: 8 + n }))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The stored shape of tensor `name`.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries.get(name).ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))
    }

    /// `len` bytes at `offset` within tensor data.
    fn read_at(&self, name: &str, offset: u64, bytes: &mut [u8]) -> Result<()> {
        let mut f = self.file.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let at = self.data_start.checked_add(offset).ok_or_else(|| Error::Model(format!("{name}: offset overflows")))?;
        f.seek(SeekFrom::Start(at)).map_err(|err| Error::Model(format!("{name}: {err}")))?;
        f.read_exact(bytes).map_err(|err| Error::Model(format!("{name}: {err}")))
    }

    /// Tensor `name` as stored, on the CPU.
    pub fn read(&self, name: &str) -> Result<Tensor> {
        let e = self.entry(name)?;
        // checked against the file's length in `open`
        let len = usize::try_from(e.end.saturating_sub(e.start)).map_err(|_| Error::Model(format!("{name}: too large")))?;
        let mut bytes = vec![0u8; len];
        self.read_at(name, e.start, &mut bytes)?;
        Ok(Tensor::from_vec(f32s(&bytes), e.shape.as_slice(), &Device::Cpu)?)
    }

    /// Rows `ids` of the 2-D tensor `name` (an embedding table), `[ids.len(), cols]` on the CPU,
    /// read positionally: only those rows' bytes leave the file. An id past the table is an error.
    pub fn read_rows(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        let e = self.entry(name)?;
        let &[rows, cols] = e.shape.as_slice() else {
            return Err(Error::Model(format!("{name}: not a 2-D table (shape {:?})", e.shape)));
        };
        if ids.len() > MAX_ROWS {
            return Err(Error::Model(format!("{name}: {} rows asked for at once (at most {MAX_ROWS})", ids.len())));
        }
        // `open` checked rows × cols × 4 fits the file, so these products cannot overflow
        let row_bytes = cols.checked_mul(4).ok_or_else(|| Error::Model(format!("{name}: row too large")))?;
        let mut out = Vec::with_capacity(ids.len().saturating_mul(cols));
        let mut buf = vec![0u8; row_bytes];
        for &id in ids {
            let id = usize::try_from(id).map_err(|_| Error::Model(format!("{name}: id {id} out of range")))?;
            if id >= rows {
                return Err(Error::Model(format!("{name}: id {id} past the table's {rows} rows")));
            }
            let offset = (id as u64).checked_mul(row_bytes as u64).and_then(|o| o.checked_add(e.start));
            let offset = offset.ok_or_else(|| Error::Model(format!("{name}: offset overflows")))?;
            self.read_at(name, offset, &mut buf)?;
            out.extend(f32s(&buf));
        }
        Ok(Tensor::from_vec(out, (ids.len(), cols), &Device::Cpu)?)
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// [`Weights`] as a candle `VarBuilder` backend.
pub struct Backend(pub Arc<Weights>);

impl SimpleBackend for Backend {
    fn get(&self, s: Shape, name: &str, _h: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
        if t.shape() != &s {
            candle_core::bail!("shape mismatch for {name}: expected {s:?}, got {:?}", t.shape())
        }
        Ok(t)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.0.read(name).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        t.to_dtype(dtype)?.to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file: the 8-byte header length, the JSON header, then `data`.
    fn write(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lc-embed-weights-{name}-{}.safetensors", std::process::id()));
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        b.extend_from_slice(data);
        std::fs::write(&p, b).unwrap();
        p
    }

    fn le(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn header_parses_and_tensors_read_back() {
        // a 3 × 2 table and a 2-vector, with metadata
        let table = [0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0];
        let mut data = le(&table);
        data.extend(le(&[7.0, 8.0]));
        let h = r#"{"__metadata__":{"format":"pt"},"t":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]},"b":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}"#;
        let p = write("ok", h, &data);
        let w = Weights::open(&p).unwrap();
        assert!(w.contains("t") && w.contains("b") && !w.contains("__metadata__"));
        assert_eq!(w.shape("t"), Some(&[3usize, 2][..]));
        assert_eq!(w.read("t").unwrap().to_vec2::<f32>().unwrap(), vec![vec![0.0, 1.0], vec![10.0, 11.0], vec![20.0, 21.0]]);
        assert_eq!(w.read("b").unwrap().to_vec1::<f32>().unwrap(), vec![7.0, 8.0]);
        // positional row reads, repeated and out of order
        let r = w.read_rows("t", &[2, 0, 2]).unwrap();
        assert_eq!(r.to_vec2::<f32>().unwrap(), vec![vec![20.0, 21.0], vec![0.0, 1.0], vec![20.0, 21.0]]);
        assert_eq!(w.read_rows("t", &[]).unwrap().dims(), &[0, 2]);
        // an id past the table, a 1-D tensor and a missing name are errors
        assert!(w.read_rows("t", &[3]).unwrap_err().to_string().contains("past the table"));
        assert!(w.read_rows("t", &[u32::MAX]).is_err());
        assert!(w.read_rows("b", &[0]).is_err());
        assert!(w.read_rows("nope", &[0]).is_err() && w.read("nope").is_err());
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn hostile_headers_are_errors() {
        let ok = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let cases = [
            // a truncated download: the header promises bytes that aren't there
            ("short", ok, 8),
            // huge shapes and offsets beyond u64
            ("huge", r#"{"a":{"dtype":"F32","shape":[1073741824,1073741824],"data_offsets":[0,4611686018427387904]}}"#, 16),
            ("wrap", r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[18446744073709551612,18446744073709551616]}}"#, 16),
            // the size does not match the shape, a dtype this crate does not read, bad offsets, not JSON
            ("size", r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,12]}}"#, 16),
            ("dtype", r#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#, 16),
            ("offs", r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0]}}"#, 16),
            ("json", "{not json", 16),
        ];
        for (name, h, n) in cases {
            let p = write(name, h, &vec![0u8; n]);
            assert!(Weights::open(&p).is_err(), "{name}");
            let _ = std::fs::remove_file(p);
        }
        let short = write("msg", ok, &[0u8; 8]);
        assert!(Weights::open(&short).err().unwrap().to_string().contains("outside the file"));
        // a header longer than the file, and a file shorter than the length prefix
        let p = std::env::temp_dir().join(format!("lc-embed-weights-hdr-{}.safetensors", std::process::id()));
        std::fs::write(&p, 1000u64.to_le_bytes()).unwrap();
        assert!(Weights::open(&p).is_err());
        std::fs::write(&p, [1u8, 2]).unwrap();
        assert!(Weights::open(&p).is_err());
        for f in [short, p] {
            let _ = std::fs::remove_file(f);
        }
    }
}
