//! Where VERA's sidecar files live: the digests, the ignored pairs, the embeddings, the keyword matrix and the review
//! queue (`docs/vera.md` → Files beside the library).
//!
//! - A library on disk (the desktop app, the CLI, MCP): its folder, through `std::fs` exactly as before ([`Sidecars::Dir`]):
//!   appends to an open file, atomic rewrites through a `.part` file and a rename, streamed reads of a large file.
//! - A library whose files live in a [`Store`] (the browser build: OPFS or IndexedDB behind the host's memory mirror,
//!   `apps/lightcraft-web/src/files.rs`): the same store as `prefs.json` and `view.json` ([`Sidecars::Store`]). New
//!   records go through [`Store::append`], which the host flushes as a real append (only the new bytes), and a
//!   compaction is a rare whole [`Store::write_atomic`].
//! - No library: memory only ([`Sidecars::Memory`]); nothing is written.
//!
//! Every file is rebuildable or a small user decision, so a read that fails is treated as a missing file and a write
//! that fails is returned for the caller to log and retry.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use lightcraft_catalog::Store;

/// A library's file store, shared between the library (`prefs.json`, `view.json`, …) and VERA's sidecars.
pub(crate) type SharedStore = Arc<Mutex<Box<dyn Store>>>;

/// Where the sidecar files of the library in use are kept.
#[derive(Clone, Default)]
pub(crate) enum Sidecars {
    /// No library (or one that keeps nothing): memory only.
    #[default]
    Memory,
    /// The library folder on disk.
    Dir(PathBuf),
    /// The library's own file store (browser storage).
    Store(SharedStore),
}

/// What reading a sidecar file found.
pub(crate) enum Loaded {
    /// No file, or one that could not be read (treated alike: the file is rebuildable).
    Missing,
    /// Over the reader's cap: ignored and rebuilt.
    TooLarge,
    Bytes(Vec<u8>),
}

/// A sidecar file opened for a streamed read.
pub(crate) enum Opened {
    Missing,
    TooLarge,
    Reader(Box<dyn Read>),
}

impl Sidecars {
    /// Both name the same place (the same folder, or the same store), so files opened from one stay valid for the other.
    pub fn same_place(&self, other: &Sidecars) -> bool {
        match (self, other) {
            (Sidecars::Memory, Sidecars::Memory) => true,
            (Sidecars::Dir(a), Sidecars::Dir(b)) => a == b,
            (Sidecars::Store(a), Sidecars::Store(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Files are written somewhere (not memory only).
    pub fn persists(&self) -> bool {
        !matches!(self, Sidecars::Memory)
    }

    /// `name`'s path, for a library on disk.
    pub fn path_of(&self, name: &str) -> Option<PathBuf> {
        match self {
            Sidecars::Dir(d) => Some(d.join(name)),
            _ => None,
        }
    }

    /// Where `name` is, for messages and `vera.status`: its path, or the store it is in.
    pub fn describe(&self, name: &str) -> String {
        match self {
            Sidecars::Memory => format!("{name} (in memory)"),
            Sidecars::Dir(d) => d.join(name).display().to_string(),
            Sidecars::Store(st) => format!("{name} in {}", lock(st).describe()),
        }
    }

    /// The size of `name`, when there is such a file.
    pub fn len(&self, name: &str) -> Option<u64> {
        match self {
            Sidecars::Memory => None,
            Sidecars::Dir(d) => std::fs::metadata(d.join(name)).ok().map(|m| m.len()),
            Sidecars::Store(st) => lock(st).read(name).ok().flatten().map(|b| b.len() as u64),
        }
    }

    /// Whether `name` exists.
    pub fn exists(&self, name: &str) -> bool {
        match self {
            Sidecars::Memory => false,
            Sidecars::Dir(d) => d.join(name).is_file(),
            Sidecars::Store(st) => lock(st).read(name).ok().flatten().is_some(),
        }
    }

    /// The whole of `name`, if it is at most `max` bytes.
    pub fn read(&self, name: &str, max: u64) -> Loaded {
        match self {
            Sidecars::Memory => Loaded::Missing,
            Sidecars::Dir(d) => {
                let path = d.join(name);
                let Ok(meta) = std::fs::metadata(&path) else { return Loaded::Missing };
                if meta.len() > max {
                    return Loaded::TooLarge;
                }
                let mut buf = Vec::new();
                // capped again while reading: the file may have grown since
                match std::fs::File::open(&path).and_then(|f| f.take(max.saturating_add(1)).read_to_end(&mut buf)) {
                    Ok(_) if buf.len() as u64 > max => Loaded::TooLarge,
                    Ok(_) => Loaded::Bytes(buf),
                    Err(_) => Loaded::Missing,
                }
            }
            Sidecars::Store(st) => match lock(st).read(name) {
                Ok(Some(b)) if b.len() as u64 > max => Loaded::TooLarge,
                Ok(Some(b)) => Loaded::Bytes(b),
                Ok(None) => Loaded::Missing,
                Err(e) => {
                    log::debug!("vera: reading {name}: {e}");
                    Loaded::Missing
                }
            },
        }
    }

    /// `name` opened for a streamed read (a large file is never held whole on the desktop), if it is at most `max` bytes.
    pub fn open(&self, name: &str, max: u64) -> Opened {
        match self {
            Sidecars::Dir(d) => {
                let path = d.join(name);
                let Ok(meta) = std::fs::metadata(&path) else { return Opened::Missing };
                if meta.len() > max {
                    return Opened::TooLarge;
                }
                match std::fs::File::open(&path) {
                    Ok(f) => Opened::Reader(Box::new(io::BufReader::new(f))),
                    Err(_) => Opened::Missing,
                }
            }
            _ => match self.read(name, max) {
                Loaded::Missing => Opened::Missing,
                Loaded::TooLarge => Opened::TooLarge,
                Loaded::Bytes(b) => Opened::Reader(Box::new(io::Cursor::new(b))),
            },
        }
    }

    /// Add `data` at the end of `name`, an existing file this format already wrote. On the desktop a file that went
    /// away is an error (never a file without its header): the caller then writes the file whole.
    pub fn append(&self, name: &str, data: &[u8]) -> io::Result<()> {
        match self {
            Sidecars::Memory => Ok(()),
            Sidecars::Dir(d) => std::fs::OpenOptions::new().append(true).open(d.join(name)).and_then(|mut f| f.write_all(data)),
            Sidecars::Store(st) => lock(st).append(name, data),
        }
    }

    /// Replace `name` with `data` atomically (desktop: a `.part` file renamed over it).
    pub fn write_atomic(&self, name: &str, data: &[u8]) -> Result<(), String> {
        match self {
            Sidecars::Memory => Ok(()),
            Sidecars::Dir(d) => crate::denoise::write_atomic(&d.join(name), data),
            Sidecars::Store(st) => lock(st).write_atomic(name, data).map_err(|e| format!("could not write {name}: {e}")),
        }
    }

    /// Replace `name` atomically with what `fill` writes, streamed (desktop: into `<name>` with extension `part_ext`,
    /// fsynced, then renamed over it, so a failed write never leaves half a file).
    pub fn write_streamed(&self, name: &str, part_ext: &str, fill: &mut dyn FnMut(&mut dyn Write) -> io::Result<()>) -> io::Result<()> {
        match self {
            Sidecars::Memory => Ok(()),
            Sidecars::Dir(d) => {
                let path = d.join(name);
                let part = path.with_extension(part_ext);
                {
                    let mut w = io::BufWriter::new(std::fs::File::create(&part)?);
                    fill(&mut w)?;
                    w.flush()?;
                    w.get_ref().sync_all()?;
                }
                std::fs::rename(&part, &path).inspect_err(|_| {
                    let _ = std::fs::remove_file(&part);
                })
            }
            Sidecars::Store(st) => lock(st).write_atomic_with(name, fill).map(|_| ()),
        }
    }

    /// Delete `name`; `Ok(true)` when there was a file. A store has no delete: the file is emptied, which every reader
    /// takes for a file that is not ours (and rebuilds).
    pub fn remove(&self, name: &str) -> io::Result<bool> {
        match self {
            Sidecars::Memory => Ok(false),
            Sidecars::Dir(d) => match std::fs::remove_file(d.join(name)) {
                Ok(()) => Ok(true),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(e),
            },
            Sidecars::Store(st) => {
                let mut st = lock(st);
                if st.read(name)?.is_none_or(|b| b.is_empty()) {
                    return Ok(false);
                }
                st.write_atomic(name, &[])?;
                Ok(true)
            }
        }
    }
}

fn lock(st: &SharedStore) -> MutexGuard<'_, Box<dyn Store>> {
    st.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where the sidecars of the library open in `s` live: its folder when it is on disk, else its file store; memory only
/// without a library.
pub(crate) fn of(s: &crate::Session) -> Sidecars {
    match &s.library {
        None => Sidecars::Memory,
        Some(l) if l.on_disk => Sidecars::Dir(l.dir.clone()),
        Some(l) => Sidecars::Store(l.shared_files()),
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! An in-memory [`Store`] that counts what was asked of it, standing in for the browser's `Files`.

    use std::collections::BTreeMap;
    use std::io;
    use std::sync::{Arc, Mutex};

    use lightcraft_catalog::Store;

    #[derive(Default)]
    pub struct Inner {
        pub files: BTreeMap<String, Vec<u8>>,
        /// (file, bytes) of every append, and the files written whole, in order.
        pub appends: Vec<(String, usize)>,
        pub rewrites: Vec<String>,
        /// Writes fail while set (a full quota).
        pub fail: bool,
    }

    /// Clones share the same files.
    #[derive(Clone, Default)]
    pub struct FakeFiles(pub Arc<Mutex<Inner>>);

    impl FakeFiles {
        pub fn get(&self, name: &str) -> Option<Vec<u8>> {
            self.0.lock().unwrap().files.get(name).cloned()
        }
        pub fn appends(&self, name: &str) -> usize {
            self.0.lock().unwrap().appends.iter().filter(|(n, _)| n == name).count()
        }
        pub fn rewrites(&self, name: &str) -> usize {
            self.0.lock().unwrap().rewrites.iter().filter(|n| *n == name).count()
        }
        pub fn set_fail(&self, fail: bool) {
            self.0.lock().unwrap().fail = fail;
        }
        /// The same files behind a fresh handle (a reload of the page).
        pub fn reloaded(&self) -> FakeFiles {
            let files = self.0.lock().unwrap().files.clone();
            FakeFiles(Arc::new(Mutex::new(Inner { files, ..Inner::default() })))
        }
        pub fn shared(&self) -> super::SharedStore {
            Arc::new(Mutex::new(Box::new(self.clone())))
        }
    }

    impl Store for FakeFiles {
        fn read(&mut self, name: &str) -> io::Result<Option<Vec<u8>>> {
            Ok(self.get(name))
        }
        fn write_atomic(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
            let mut g = self.0.lock().unwrap();
            if g.fail {
                return Err(io::Error::other("QuotaExceededError"));
            }
            g.files.insert(name.to_string(), data.to_vec());
            g.rewrites.push(name.to_string());
            Ok(())
        }
        fn append(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
            let mut g = self.0.lock().unwrap();
            if g.fail {
                return Err(io::Error::other("QuotaExceededError"));
            }
            g.files.entry(name.to_string()).or_default().extend_from_slice(data);
            g.appends.push((name.to_string(), data.len()));
            Ok(())
        }
        fn truncate(&mut self, name: &str, len: u64) -> io::Result<()> {
            if let Some(f) = self.0.lock().unwrap().files.get_mut(name) {
                f.truncate(len as usize);
            }
            Ok(())
        }
        fn describe(&self) -> String {
            "fake browser storage".into()
        }
    }
}
