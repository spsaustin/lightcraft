//! RFC 3339 timestamps: the strict shape the conformance profile asserts (errata E23), parsing
//! to an instant for chronology rules, and formatting the current time.

/// Days from 1970-01-01 to `y-m-d` (proleptic Gregorian; Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        _ => 28,
    }
}

fn digits(b: &[u8]) -> Option<i64> {
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    b.iter().try_fold(0i64, |acc, &c| acc.checked_mul(10)?.checked_add(i64::from(c - b'0')))
}

/// A parsed timestamp: seconds since the epoch (UTC), nanoseconds, and whether it was written in UTC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Instant {
    pub secs: i64,
    pub nanos: u32,
}

/// Parse `s` as the profile's RFC 3339 `date-time`: `YYYY-MM-DDTHH:MM:SS[.frac](Z|±HH:MM)`, with
/// real calendar values. Returns the instant and whether the designator was `Z`.
pub(crate) fn parse(s: &str) -> Option<(Instant, bool)> {
    let b = s.as_bytes();
    if b.len() < 20 {
        return None;
    }
    let (y, mo, d) = (digits(b.get(0..4)?)?, digits(b.get(5..7)?)?, digits(b.get(8..10)?)?);
    if b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b't') || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let (h, mi, se) = (digits(b.get(11..13)?)?, digits(b.get(14..16)?)?, digits(b.get(17..19)?)?);
    if !(1..=12).contains(&mo) || d < 1 || d > days_in_month(y, mo) || h > 23 || mi > 59 || se > 59 || y < 1 {
        return None;
    }
    let mut p = 19;
    let mut nanos = 0u32;
    if b.get(p) == Some(&b'.') {
        let start = p + 1;
        let mut q = start;
        while b.get(q).is_some_and(u8::is_ascii_digit) {
            q += 1;
        }
        if q == start {
            return None;
        }
        let frac = b.get(start..q)?;
        let mut n = 0u32;
        for i in 0..9 {
            n = n * 10 + frac.get(i).map(|c| u32::from(c - b'0')).unwrap_or(0);
        }
        nanos = n;
        p = q;
    }
    let (offset, utc) = match b.get(p..)? {
        [b'Z' | b'z'] => (0i64, true),
        [sign @ (b'+' | b'-'), h1, h2, b':', m1, m2] => {
            let oh = digits(&[*h1, *h2])?;
            let om = digits(&[*m1, *m2])?;
            if oh > 23 || om > 59 {
                return None;
            }
            let off = oh * 3600 + om * 60;
            (if *sign == b'-' { -off } else { off }, false)
        }
        _ => return None,
    };
    let secs = days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se - offset;
    Some((Instant { secs, nanos }, utc))
}

/// `YYYY-MM-DD` with real calendar values.
pub(crate) fn valid_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    match (digits(&b[0..4]), digits(&b[5..7]), digits(&b[8..10])) {
        (Some(y), Some(m), Some(d)) => y >= 1 && (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m),
        _ => false,
    }
}

/// The current time as RFC 3339 UTC with second precision.
pub fn now_rfc3339() -> String {
    let secs = web_time::SystemTime::now().duration_since(web_time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    format_utc(secs)
}

/// `secs` since the epoch as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn format_utc(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_rejects() {
        let (a, utc) = parse("2026-08-21T14:02:11Z").expect("valid");
        assert!(utc);
        assert_eq!(format_utc(a.secs), "2026-08-21T14:02:11Z");
        let (b, utc) = parse("2026-08-21T16:02:11.5+02:00").expect("valid");
        assert!(!utc);
        assert_eq!(b.secs, a.secs);
        assert_eq!(b.nanos, 500_000_000);
        for bad in ["yesterday", "2026-13-45T99:99:99", "2026-02-30T00:00:00Z", "2026-08-21 14:02:11Z", "2026-08-21T14:02:11", "2026-08-21T14:02:60Z"]
        {
            assert!(parse(bad).is_none(), "{bad}");
        }
        assert!(valid_date("2024-02-29"));
        assert!(!valid_date("2023-02-29"));
    }
}
