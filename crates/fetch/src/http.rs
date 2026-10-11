//! A small HTTP/1.1 client for model downloads (`GET`) and the Describe lane's requests to an
//! OpenAI-compatible endpoint (`POST`, [`request`]), in pure Rust: `std::net` sockets and, for
//! `https://`, rustls with the RustCrypto provider (`rustls-rustcrypto`; no `ring` or
//! `aws-lc-rs`, so nothing is compiled from C or assembly) and the Mozilla root certificates
//! (`webpki-roots`).
//!
//! Only what those need: one request per connection (`Connection: close`), a request body sent
//! with its `Content-Length`, `Range` requests for resuming, redirects (followed by the caller),
//! `Content-Length`, chunked and read-to-close bodies. Which hosts may be reached over plain
//! `http://` is the caller's policy ([`Url::is_loopback`] helps it decide). Every read waits at most `stall` for data and checks the
//! cancel flag at least twice a second, so a dead server or a cancelled download never hangs
//! the downloading thread. No proxies (see docs/ai-masks.md).

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Why a request failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// The URL can't be used (scheme, host, characters).
    BadUrl(String),
    /// Name resolution or connecting failed (or took longer than the connect timeout).
    Connect(String),
    /// No data arrived for the stall timeout.
    Stalled,
    /// The cancel flag was raised.
    Cancelled,
    /// TLS failed (certificate, handshake).
    Tls(String),
    /// The server's response could not be understood.
    Protocol(String),
    /// Any other I/O error.
    Io(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::BadUrl(e) => write!(f, "unusable URL: {e}"),
            HttpError::Connect(e) => write!(f, "could not connect: {e}"),
            HttpError::Stalled => write!(f, "the server stopped sending data"),
            HttpError::Cancelled => write!(f, "cancelled"),
            HttpError::Tls(e) => write!(f, "secure connection failed: {e}"),
            HttpError::Protocol(e) => write!(f, "unexpected response: {e}"),
            HttpError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// A parsed `http://` or `https://` URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub tls: bool,
    pub host: String,
    pub port: u16,
    /// Path and query, starting with `/`.
    pub path: String,
}

impl Url {
    pub fn parse(s: &str) -> Result<Url, HttpError> {
        let bad = |why: &str| HttpError::BadUrl(why.to_string());
        if s.len() > 16 * 1024 {
            return Err(bad("URL exceeds 16 KiB"));
        }
        let s = s.trim();
        if s.chars().any(|c| c.is_control() || c == ' ') {
            return Err(bad("contains spaces or control characters"));
        }
        let (tls, rest) = if let Some(r) = s.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = s.strip_prefix("http://") {
            (false, r)
        } else {
            return Err(bad("only http:// and https:// are supported"));
        };
        let rest = rest.split('#').next().unwrap_or_default();
        let (authority, path) = match rest.find(['/', '?']) {
            Some(i) => (rest.get(..i).unwrap_or_default(), rest.get(i..).unwrap_or_default()),
            None => (rest, ""),
        };
        let path = if path.starts_with('/') { path.to_string() } else { format!("/{path}") };
        if authority.contains('@') {
            return Err(bad("user names and passwords in URLs are not supported"));
        }
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let end = v6.find(']').ok_or_else(|| bad("unterminated IPv6 address"))?;
            let host = v6.get(..end).unwrap_or_default();
            let after = v6.get(end + 1..).unwrap_or_default();
            (host.to_string(), after.strip_prefix(':'))
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), Some(p)),
                None => (authority.to_string(), None),
            }
        };
        let port = match port {
            Some(p) => p.parse::<u16>().ok().filter(|p| *p != 0).ok_or_else(|| bad("bad port"))?,
            None if tls => 443,
            None => 80,
        };
        if host.is_empty() || !host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | ':' | '_')) {
            return Err(bad("bad host name"));
        }
        Ok(Url { tls, host: host.to_ascii_lowercase(), port, path })
    }

    /// `Location` of a redirect, resolved against this URL.
    pub fn join(&self, location: &str) -> Result<Url, HttpError> {
        let l = location.trim();
        if l.starts_with("http://") || l.starts_with("https://") {
            return Url::parse(l);
        }
        // another scheme (`ftp:`, `file:`, `data:`…)
        if let Some((scheme, _)) = l.split_once(':')
            && !scheme.is_empty()
            && scheme.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        {
            return Err(HttpError::BadUrl(format!("redirect to an unsupported location `{scheme}:`")));
        }
        let scheme = if self.tls { "https" } else { "http" };
        let host = if self.host.contains(':') { format!("[{}]", self.host) } else { self.host.clone() };
        let base = format!("{scheme}://{host}:{}", self.port);
        if let Some(rest) = l.strip_prefix("//") {
            return Url::parse(&format!("{scheme}://{rest}"));
        }
        if l.starts_with('/') {
            return Url::parse(&format!("{base}{l}"));
        }
        // relative to the current directory
        let dir = self.path.split('?').next().unwrap_or("/");
        let dir = dir.rsplit_once('/').map_or("", |(d, _)| d);
        Url::parse(&format!("{base}{dir}/{l}"))
    }

    fn host_header(&self) -> String {
        let host = if self.host.contains(':') { format!("[{}]", self.host) } else { self.host.clone() };
        let default = if self.tls { 443 } else { 80 };
        if self.port == default { host } else { format!("{host}:{}", self.port) }
    }

    /// Whether the host is this computer: `localhost`, an address in 127.0.0.0/8, or `::1`. The engine allows plain
    /// `http://` only to such a host (anything else needs `https://`).
    pub fn is_loopback(&self) -> bool {
        if self.host == "localhost" {
            return true;
        }
        self.host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
    }

    /// The URL without its query (for messages: a query may hold a signed token).
    pub fn display(&self) -> String {
        let scheme = if self.tls { "https" } else { "http" };
        format!("{scheme}://{}{}", self.host_header(), self.path.split('?').next().unwrap_or("/"))
    }
}

/// Timeouts and the cancel flag for one request.
pub struct Limits<'a> {
    pub connect: Duration,
    pub stall: Duration,
    pub cancel: &'a AtomicBool,
}

/// How often a blocked read wakes up to check the cancel flag.
const TICK: Duration = Duration::from_millis(250);
/// Longest status line + headers accepted.
const MAX_HEAD: usize = 64 * 1024;

enum Stream {
    Plain(TcpStream),
    Tls(Box<rustls::ClientConnection>, TcpStream),
}

fn is_timeout(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted)
}

fn tls_config() -> Result<Arc<rustls::ClientConfig>, HttpError> {
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls_rustcrypto::provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| HttpError::Tls(e.to_string()))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

impl Stream {
    fn connect(url: &Url, limits: &Limits) -> Result<Stream, HttpError> {
        let addrs: Vec<_> = (url.host.as_str(), url.port).to_socket_addrs().map_err(|e| HttpError::Connect(format!("{}: {e}", url.host)))?.collect();
        if addrs.is_empty() {
            return Err(HttpError::Connect(format!("{}: no address", url.host)));
        }
        let mut last = String::new();
        let mut sock = None;
        for a in addrs {
            if limits.cancel.load(Ordering::Relaxed) {
                return Err(HttpError::Cancelled);
            }
            match TcpStream::connect_timeout(&a, limits.connect) {
                Ok(s) => {
                    sock = Some(s);
                    break;
                }
                Err(e) => last = format!("{}: {e}", url.host),
            }
        }
        let sock = sock.ok_or(HttpError::Connect(last))?;
        sock.set_read_timeout(Some(TICK)).map_err(|e| HttpError::Io(e.to_string()))?;
        sock.set_write_timeout(Some(TICK)).map_err(|e| HttpError::Io(e.to_string()))?;
        let _ = sock.set_nodelay(true);
        if !url.tls {
            return Ok(Stream::Plain(sock));
        }
        let name = rustls::pki_types::ServerName::try_from(url.host.clone()).map_err(|e| HttpError::Tls(format!("{}: {e}", url.host)))?;
        let conn = rustls::ClientConnection::new(tls_config()?, name).map_err(|e| HttpError::Tls(e.to_string()))?;
        let mut s = Stream::Tls(Box::new(conn), sock);
        s.handshake(limits)?;
        Ok(s)
    }

    /// Wait for `f` to make progress, retrying timeouts until `stall` passes without any.
    fn retry<T>(limits: &Limits, mut f: impl FnMut() -> std::io::Result<T>) -> Result<T, HttpError> {
        let start = Instant::now();
        loop {
            if limits.cancel.load(Ordering::Relaxed) {
                return Err(HttpError::Cancelled);
            }
            match f() {
                Ok(v) => return Ok(v),
                Err(e) if is_timeout(&e) => {
                    if start.elapsed() >= limits.stall {
                        return Err(HttpError::Stalled);
                    }
                }
                Err(e) if e.kind() == ErrorKind::InvalidData => return Err(HttpError::Tls(e.to_string())),
                Err(e) => return Err(HttpError::Io(e.to_string())),
            }
        }
    }

    fn handshake(&mut self, limits: &Limits) -> Result<(), HttpError> {
        let Stream::Tls(conn, sock) = self else { return Ok(()) };
        while conn.is_handshaking() {
            Self::retry(limits, || conn.complete_io(sock))?;
        }
        Ok(())
    }

    fn write_all(&mut self, data: &[u8], limits: &Limits) -> Result<(), HttpError> {
        match self {
            Stream::Plain(sock) => {
                let mut rest = data;
                while !rest.is_empty() {
                    let n = Self::retry(limits, || sock.write(rest))?;
                    if n == 0 {
                        return Err(HttpError::Io("connection closed while sending the request".into()));
                    }
                    rest = rest.get(n..).unwrap_or_default();
                }
                Ok(())
            }
            Stream::Tls(conn, sock) => {
                // buffered by rustls, then sent as TLS records
                conn.writer().write_all(data).map_err(|e| HttpError::Tls(e.to_string()))?;
                while conn.wants_write() {
                    Self::retry(limits, || conn.write_tls(sock))?;
                }
                Ok(())
            }
        }
    }

    /// Read some bytes (0 at the end of the stream), waiting at most `stall` for them.
    fn read(&mut self, buf: &mut [u8], limits: &Limits) -> Result<usize, HttpError> {
        match self {
            Stream::Plain(sock) => Self::retry(limits, || sock.read(buf)),
            Stream::Tls(conn, sock) => {
                let start = Instant::now();
                loop {
                    if limits.cancel.load(Ordering::Relaxed) {
                        return Err(HttpError::Cancelled);
                    }
                    match conn.reader().read(buf) {
                        Ok(n) => return Ok(n),
                        // closed without close_notify: the caller checks the length it got
                        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(0),
                        Err(e) if e.kind() == ErrorKind::WouldBlock => {}
                        Err(e) => return Err(HttpError::Tls(e.to_string())),
                    }
                    // no plaintext buffered: more TLS records from the socket
                    match conn.read_tls(sock) {
                        Ok(0) => return Ok(0),
                        Ok(_) => {
                            conn.process_new_packets().map_err(|e| HttpError::Tls(e.to_string()))?;
                            while conn.wants_write() {
                                Self::retry(limits, || conn.write_tls(sock))?;
                            }
                        }
                        Err(e) if is_timeout(&e) => {
                            if start.elapsed() >= limits.stall {
                                return Err(HttpError::Stalled);
                            }
                        }
                        Err(e) => return Err(HttpError::Io(e.to_string())),
                    }
                }
            }
        }
    }
}

/// The connection with a read buffer (headers and chunk sizes are read line by line).
struct Conn {
    stream: Stream,
    buf: Vec<u8>,
    pos: usize,
}

impl Conn {
    fn fill(&mut self, limits: &Limits) -> Result<usize, HttpError> {
        if self.pos >= self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        }
        let mut tmp = [0u8; 64 * 1024];
        let n = self.stream.read(&mut tmp, limits)?;
        self.buf.extend_from_slice(tmp.get(..n).unwrap_or_default());
        Ok(n)
    }

    /// One line without its CRLF; at most `max` bytes.
    fn line(&mut self, max: usize, limits: &Limits) -> Result<String, HttpError> {
        loop {
            let avail = self.buf.get(self.pos..).unwrap_or_default();
            if let Some(i) = avail.iter().position(|b| *b == b'\n') {
                let line = avail.get(..i).unwrap_or_default();
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let s = String::from_utf8_lossy(line).into_owned();
                self.pos = self.pos.saturating_add(i + 1);
                return Ok(s);
            }
            if avail.len() > max {
                return Err(HttpError::Protocol("header line too long".into()));
            }
            // keep the partial line, read more after it
            if self.pos > 0 {
                self.buf.drain(..self.pos.min(self.buf.len()));
                self.pos = 0;
            }
            let mut tmp = [0u8; 16 * 1024];
            let n = self.stream.read(&mut tmp, limits)?;
            if n == 0 {
                return Err(HttpError::Protocol("connection closed in the middle of the response".into()));
            }
            self.buf.extend_from_slice(tmp.get(..n).unwrap_or_default());
        }
    }

    /// Up to `out.len()` body bytes (0 at the end of the stream).
    fn read(&mut self, out: &mut [u8], limits: &Limits) -> Result<usize, HttpError> {
        if self.pos >= self.buf.len() && self.fill(limits)? == 0 {
            return Ok(0);
        }
        let avail = self.buf.get(self.pos..).unwrap_or_default();
        let n = avail.len().min(out.len());
        if let (Some(dst), Some(src)) = (out.get_mut(..n), avail.get(..n)) {
            dst.copy_from_slice(src);
        }
        self.pos = self.pos.saturating_add(n);
        Ok(n)
    }
}

/// A response's status, headers and body.
pub struct Response {
    pub status: u16,
    headers: Vec<(String, String)>,
    conn: Conn,
    body: BodyMode,
}

enum BodyMode {
    Length(u64),
    Chunked { left: u64, done: bool },
    Close,
    Done,
}

impl Response {
    /// The first header named `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The body's length when the server announced it.
    pub fn content_length(&self) -> Option<u64> {
        match self.body {
            BodyMode::Length(n) => Some(n),
            _ => None,
        }
    }

    /// The whole body, refused once it passes `cap` bytes (so a hostile server cannot make it grow without bound).
    pub fn read_to_end(&mut self, cap: usize, limits: &Limits) -> Result<Vec<u8>, HttpError> {
        if self.content_length().is_some_and(|n| n > cap as u64) {
            return Err(HttpError::Protocol(format!("the response is larger than {cap} bytes")));
        }
        let mut out = Vec::new();
        let mut buf = [0u8; 16 * 1024];
        loop {
            let n = self.read(&mut buf, limits)?;
            if n == 0 {
                return Ok(out);
            }
            if out.len().saturating_add(n) > cap {
                return Err(HttpError::Protocol(format!("the response is larger than {cap} bytes")));
            }
            out.extend_from_slice(buf.get(..n).unwrap_or_default());
        }
    }

    /// Read body bytes into `out` (0 at the end of the body).
    pub fn read(&mut self, out: &mut [u8], limits: &Limits) -> Result<usize, HttpError> {
        match &mut self.body {
            BodyMode::Done => Ok(0),
            BodyMode::Length(left) => {
                if *left == 0 {
                    self.body = BodyMode::Done;
                    return Ok(0);
                }
                let want = usize::try_from(*left).unwrap_or(usize::MAX).min(out.len());
                let n = self.conn.read(out.get_mut(..want).unwrap_or_default(), limits)?;
                if n == 0 {
                    return Err(HttpError::Protocol("the connection closed before the end of the file".into()));
                }
                *left = left.saturating_sub(n as u64);
                Ok(n)
            }
            BodyMode::Close => {
                let n = self.conn.read(out, limits)?;
                if n == 0 {
                    self.body = BodyMode::Done;
                }
                Ok(n)
            }
            BodyMode::Chunked { left, done } => {
                if *done {
                    return Ok(0);
                }
                if *left == 0 {
                    let line = self.conn.line(1024, limits)?;
                    let hex = line.split(';').next().unwrap_or_default().trim();
                    let size = u64::from_str_radix(hex, 16).map_err(|_| HttpError::Protocol(format!("bad chunk size `{hex}`")))?;
                    if size == 0 {
                        // trailers, up to the empty line
                        let mut n = 0;
                        while !self.conn.line(MAX_HEAD, limits)?.is_empty() {
                            n += 1;
                            if n > 100 {
                                return Err(HttpError::Protocol("too many trailers".into()));
                            }
                        }
                        *done = true;
                        return Ok(0);
                    }
                    *left = size;
                }
                let want = usize::try_from(*left).unwrap_or(usize::MAX).min(out.len());
                let n = self.conn.read(out.get_mut(..want).unwrap_or_default(), limits)?;
                if n == 0 {
                    return Err(HttpError::Protocol("the connection closed in the middle of a chunk".into()));
                }
                *left = left.saturating_sub(n as u64);
                if *left == 0 && !self.conn.line(16, limits)?.is_empty() {
                    return Err(HttpError::Protocol("missing CRLF after a chunk".into()));
                }
                Ok(n)
            }
        }
    }
}

/// Send `GET url` with extra `headers` and read the status line and headers.
pub fn get(url: &Url, headers: &[(&str, String)], limits: &Limits) -> Result<Response, HttpError> {
    request("GET", url, headers, None, limits)
}

/// Bodies sent are at most this large (a request is held in memory whole).
pub const MAX_REQUEST_BODY: usize = 64 << 20;

/// Send `method url` with extra `headers` and an optional `body`, and read the status line and headers. A body is
/// sent with its `Content-Length` (and `Content-Type: application/octet-stream` unless `headers` name one); the same
/// connect, stall and cancel limits apply to sending it as to reading the response. `method` is an upper-case token
/// (`GET`, `POST`, …). Header names and values may not hold line breaks.
pub fn request(method: &str, url: &Url, headers: &[(&str, String)], body: Option<&[u8]>, limits: &Limits) -> Result<Response, HttpError> {
    if method.is_empty() || method.len() > 16 || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::BadUrl(format!("unsupported method `{}`", method.chars().take(16).collect::<String>())));
    }
    if body.is_some_and(|b| b.len() > MAX_REQUEST_BODY) {
        return Err(HttpError::BadUrl(format!("a request body is at most {} MiB", MAX_REQUEST_BODY >> 20)));
    }
    let mut req = format!(
        "{method} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: LightCraft/{}\r\nAccept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n",
        url.path,
        url.host_header(),
        env!("CARGO_PKG_VERSION")
    );
    for (k, v) in headers {
        if v.chars().any(|c| c == '\r' || c == '\n') || k.is_empty() || k.chars().any(|c| c == '\r' || c == '\n' || c == ':' || c == ' ') {
            return Err(HttpError::BadUrl("header with a line break".into()));
        }
        if body.is_some() && k.eq_ignore_ascii_case("content-length") {
            // set from the body below; a second, different one would be a smuggling hazard
            continue;
        }
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
            req.push_str("Content-Type: application/octet-stream\r\n");
        }
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    let mut stream = Stream::connect(url, limits)?;
    stream.write_all(req.as_bytes(), limits)?;
    if let Some(b) = body {
        // in pieces, so TLS never buffers more than one piece of plaintext
        for piece in b.chunks(16 * 1024) {
            stream.write_all(piece, limits)?;
        }
    }
    let mut conn = Conn { stream, buf: Vec::new(), pos: 0 };
    let status_line = conn.line(MAX_HEAD, limits)?;
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or_default();
    if !version.starts_with("HTTP/1.") {
        return Err(HttpError::Protocol(format!("not an HTTP/1.x response: {}", status_line.chars().take(60).collect::<String>())));
    }
    let status = parts.next().and_then(|s| s.parse::<u16>().ok()).ok_or_else(|| HttpError::Protocol("bad status line".into()))?;
    let mut headers = Vec::new();
    let mut total = status_line.len();
    loop {
        let line = conn.line(MAX_HEAD, limits)?;
        if line.is_empty() {
            break;
        }
        total = total.saturating_add(line.len());
        if total > MAX_HEAD || headers.len() > 200 {
            return Err(HttpError::Protocol("response headers too large".into()));
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let find = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str());
    let body = if method == "HEAD" || status == 204 || status == 304 || (100..200).contains(&status) {
        BodyMode::Done
    } else if find("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        BodyMode::Chunked { left: 0, done: false }
    } else if let Some(n) = find("content-length") {
        BodyMode::Length(n.trim().parse::<u64>().map_err(|_| HttpError::Protocol(format!("bad Content-Length `{n}`")))?)
    } else {
        BodyMode::Close
    };
    Ok(Response { status, headers, conn, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_oversized_and_control_character_urls() {
        assert!(Url::parse(&format!("https://example.com/{}", "x".repeat(16 * 1024))).is_err());
        assert!(Url::parse("https://example.com/x?secret=abc\r\nInjected: 1").is_err());
    }

    #[test]
    fn parses_urls() {
        let u = Url::parse("https://cdn.example.com/models/sam3/model.safetensors?sig=1#x").unwrap();
        assert_eq!((u.tls, u.host.as_str(), u.port, u.path.as_str()), (true, "cdn.example.com", 443, "/models/sam3/model.safetensors?sig=1"));
        assert_eq!(u.display(), "https://cdn.example.com/models/sam3/model.safetensors");
        let u = Url::parse("http://127.0.0.1:8080").unwrap();
        assert_eq!((u.tls, u.port, u.path.as_str()), (false, 8080, "/"));
        let u = Url::parse("http://[::1]:9/a").unwrap();
        assert_eq!((u.host.as_str(), u.port), ("::1", 9));
        for bad in ["ftp://x/y", "https://", "https://a b/", "https://u:p@h/", "http://h:0/", "http://h:99999/", "https://h/\r\nX: y", "file:///etc"]
        {
            assert!(Url::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn resolves_redirects() {
        let u = Url::parse("https://a.example/x/y/file?q").unwrap();
        assert_eq!(u.join("https://b.example/z").unwrap().host, "b.example");
        assert_eq!(u.join("/root").unwrap().path, "/root");
        assert_eq!(u.join("other").unwrap().path, "/x/y/other");
        assert_eq!(u.join("//c.example/p").unwrap().host, "c.example");
        assert!(u.join("ftp://nope").is_err());
    }

    #[test]
    fn loopback_hosts() {
        for (u, want) in [
            ("http://localhost:8080/v1", true),
            ("http://127.0.0.1/", true),
            ("http://127.10.20.30:1/", true),
            ("http://[::1]:11434/", true),
            ("http://LOCALHOST/", true),
            ("https://example.com/", false),
            ("http://192.168.1.4:8080/", false),
            ("http://10.0.0.1/", false),
            ("http://localhost.example.com/", false),
            ("http://[::2]/", false),
            ("http://0.0.0.0/", false),
        ] {
            assert_eq!(Url::parse(u).unwrap().is_loopback(), want, "{u}");
        }
    }

    /// A one-request-per-connection stub on loopback: reads the request head and its `Content-Length` body, hands both
    /// to `answer`, writes what it returns and closes.
    fn stub(answer: impl Fn(&str, &[u8]) -> Vec<u8> + Send + 'static) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            for conn in listener.incoming() {
                let mut sock = conn.unwrap();
                let mut data = Vec::new();
                let mut buf = [0u8; 8192];
                let head_end = loop {
                    let n = sock.read(&mut buf).unwrap();
                    assert!(n > 0, "the client closed before sending its request");
                    data.extend_from_slice(&buf[..n]);
                    if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&data[..head_end]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                    .unwrap_or(0);
                while data.len() < head_end + len {
                    let n = sock.read(&mut buf).unwrap();
                    assert!(n > 0);
                    data.extend_from_slice(&buf[..n]);
                }
                let reply = answer(&head, &data[head_end..head_end + len]);
                let quit = head.starts_with("QUIT");
                sock.write_all(&reply).unwrap();
                drop(sock);
                if quit {
                    return;
                }
            }
        });
        (base, handle)
    }

    fn quit(base: &str, handle: std::thread::JoinHandle<()>) {
        let never = AtomicBool::new(false);
        let _ = request("QUIT", &Url::parse(base).unwrap(), &[], None, &limits(&never));
        handle.join().unwrap();
    }

    fn limits(cancel: &AtomicBool) -> Limits<'_> {
        Limits { connect: Duration::from_secs(5), stall: Duration::from_secs(5), cancel }
    }

    #[test]
    fn post_sends_the_body_with_its_length_and_type() {
        let (base, handle) = stub(|head, body| {
            let first = head.lines().next().unwrap_or_default().to_string();
            let header = |name: &str| {
                head.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.trim().to_string()))
            };
            let echo = format!(
                "{first}\n{}\n{}\n{}\n{}",
                header("content-type").unwrap_or_default(),
                header("content-length").unwrap_or_default(),
                header("authorization").unwrap_or_default(),
                body.len()
            );
            let mut r =
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{echo}", echo.len() + body.len()).into_bytes();
            r.extend_from_slice(body);
            r
        });
        let never = AtomicBool::new(false);
        let url = Url::parse(&format!("{base}/v1/chat/completions")).unwrap();
        let body = br#"{"model":"m","messages":[]}"#;
        let headers = [("Content-Type", "application/json".to_string()), ("Authorization", "Bearer abc".to_string())];
        let mut r = request("POST", &url, &headers, Some(body), &limits(&never)).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.header("content-type"), Some("text/plain"));
        let got = String::from_utf8(r.read_to_end(1 << 20, &limits(&never)).unwrap()).unwrap();
        let mut lines = got.lines();
        assert_eq!(lines.next(), Some("POST /v1/chat/completions HTTP/1.1"));
        assert_eq!(lines.next(), Some("application/json"));
        assert_eq!(lines.next(), Some(body.len().to_string().as_str()));
        assert_eq!(lines.next(), Some("Bearer abc"));
        assert!(got.ends_with(std::str::from_utf8(body).unwrap()), "{got}");
        // a large body goes in pieces and arrives whole; a caller's Content-Length is replaced by the real one
        let big: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let headers = [("Content-Length", "5".to_string())];
        let mut r = request("POST", &url, &headers, Some(&big), &limits(&never)).unwrap();
        let got = r.read_to_end(1 << 20, &limits(&never)).unwrap();
        assert!(got.ends_with(&big), "the body is echoed whole");
        let text = String::from_utf8_lossy(&got[..got.len() - big.len()]).to_string();
        assert!(text.contains("application/octet-stream") && text.contains("300000"), "{text}");
        // the response cap is enforced
        let mut r = request("POST", &url, &[], Some(&big), &limits(&never)).unwrap();
        assert!(matches!(r.read_to_end(64 * 1024, &limits(&never)), Err(HttpError::Protocol(_))));
        // GET is the same request without a body
        let mut r = get(&url, &[], &limits(&never)).unwrap();
        let got = String::from_utf8(r.read_to_end(1 << 20, &limits(&never)).unwrap()).unwrap();
        assert!(got.starts_with("GET /v1/chat/completions HTTP/1.1\n\n\n"), "no body headers on a GET: {got}");
        // hostile methods and headers are refused before connecting
        for m in ["", "post", "GET /x HTTP/1.1\r\nX:", "ABCDEFGHIJKLMNOPQ"] {
            assert!(matches!(request(m, &url, &[], None, &limits(&never)), Err(HttpError::BadUrl(_))), "{m:?}");
        }
        let evil = [("X-A", "1\r\nInjected: yes".to_string())];
        assert!(matches!(request("POST", &url, &evil, Some(b"x"), &limits(&never)), Err(HttpError::BadUrl(_))));
        let evil = [("X A", "1".to_string())];
        assert!(matches!(request("POST", &url, &evil, Some(b"x"), &limits(&never)), Err(HttpError::BadUrl(_))));
        // a raised cancel flag stops the request before it is sent
        let cancelled = AtomicBool::new(true);
        assert_eq!(request("POST", &url, &[], Some(b"x"), &limits(&cancelled)).err(), Some(HttpError::Cancelled));
        quit(&base, handle);
    }

    #[test]
    fn post_reads_a_429_with_retry_after_and_chunked_bodies() {
        let (base, handle) = stub(|head, _| {
            if head.starts_with("POST /busy") {
                b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nContent-Length: 4\r\n\r\nbusy".to_vec()
            } else if head.starts_with("POST /chunked") {
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5;x=y\r\nhello\r\n7\r\n, world\r\n0\r\nTrailer: t\r\n\r\n".to_vec()
            } else if head.starts_with("POST /close") {
                b"HTTP/1.1 500 Internal Server Error\r\n\r\nbroken".to_vec()
            } else {
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec()
            }
        });
        let never = AtomicBool::new(false);
        let url = |p: &str| Url::parse(&format!("{base}{p}")).unwrap();
        let mut r = request("POST", &url("/busy"), &[], Some(b"{}"), &limits(&never)).unwrap();
        assert_eq!((r.status, r.header("retry-after")), (429, Some("7")));
        assert_eq!(r.read_to_end(1024, &limits(&never)).unwrap(), b"busy");
        let mut r = request("POST", &url("/chunked"), &[], Some(b"{}"), &limits(&never)).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.content_length(), None);
        assert_eq!(r.read_to_end(1024, &limits(&never)).unwrap(), b"hello, world");
        // a chunked body over the cap is refused too
        let mut r = request("POST", &url("/chunked"), &[], Some(b"{}"), &limits(&never)).unwrap();
        assert!(r.read_to_end(8, &limits(&never)).is_err());
        let mut r = request("POST", &url("/close"), &[], Some(b"{}"), &limits(&never)).unwrap();
        assert_eq!(r.status, 500);
        assert_eq!(r.read_to_end(1024, &limits(&never)).unwrap(), b"broken");
        let mut r = request("HEAD", &url("/busy"), &[], None, &limits(&never)).unwrap();
        assert_eq!(r.read_to_end(1024, &limits(&never)).unwrap(), b"", "a HEAD response has no body");
        quit(&base, handle);
    }

    #[test]
    fn a_silent_server_stalls() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let keep = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_millis(1500));
            drop(sock);
        });
        let never = AtomicBool::new(false);
        let quick = Limits { connect: Duration::from_secs(5), stall: Duration::from_millis(600), cancel: &never };
        let started = Instant::now();
        assert_eq!(request("POST", &url, &[], Some(b"{}"), &quick).err(), Some(HttpError::Stalled));
        assert!(started.elapsed() < Duration::from_millis(1400));
        keep.join().unwrap();
    }

    #[test]
    fn the_tls_config_builds_with_the_pure_rust_provider() {
        assert!(tls_config().is_ok());
    }
}
