//! Object Modification Time messages: the new binary form (type 0x0012,
//! spec IV.A.2.s) and the deprecated ASCII form (type 0x000E, IV.A.2.o).

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// Object modification time as seconds since the UNIX epoch (type 0x0012).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModificationTimeMessage {
    /// Seconds after 1970-01-01T00:00:00Z.
    pub seconds: u32,
}

impl ModificationTimeMessage {
    /// Decode the message body (version 1 only; version 0 was never used).
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "modification time message version {version}"
            )));
        }
        r.skip(3)?;
        Ok(ModificationTimeMessage {
            seconds: r.read_u32()?,
        })
    }
}

/// Old-style ASCII modification time (type 0x000E), interpreted as UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModificationTimeOldMessage {
    /// Four-digit year.
    pub year: u16,
    /// Month, 1–12.
    pub month: u8,
    /// Day of month, 1–31.
    pub day: u8,
    /// Hour, 0–23.
    pub hour: u8,
    /// Minute, 0–59.
    pub minute: u8,
    /// Second, 0–59 (60 tolerated for leap seconds).
    pub second: u8,
}

fn ascii_num(digits: &[u8]) -> Result<u16> {
    let mut v: u16 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return Err(Error::invalid("non-digit in modification time"));
        }
        v = v * 10 + u16::from(d - b'0');
    }
    Ok(v)
}

impl ModificationTimeOldMessage {
    /// Decode the message body: `YYYYMMDDhhmmss` followed by 2 reserved
    /// bytes (not required to be present).
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let year = ascii_num(r.read_bytes(4)?)?;
        let mut two = || -> Result<u8> { Ok(ascii_num(r.read_bytes(2)?)? as u8) };
        let m = ModificationTimeOldMessage {
            year,
            month: two()?,
            day: two()?,
            hour: two()?,
            minute: two()?,
            second: two()?,
        };
        if !(1..=12).contains(&m.month)
            || !(1..=31).contains(&m.day)
            || m.hour > 23
            || m.minute > 59
            || m.second > 60
        {
            return Err(Error::invalid("modification time field out of range"));
        }
        Ok(m)
    }

    /// Seconds since the UNIX epoch (negative before 1970).
    pub fn to_unix_seconds(&self) -> i64 {
        let days = days_from_civil(i64::from(self.year), self.month, self.day);
        days * 86_400
            + i64::from(self.hour) * 3600
            + i64::from(self.minute) * 60
            + i64::from(self.second)
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: u8, d: u8) -> i64 {
    let (m, d) = (i64::from(m), i64::from(d));
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_style() {
        let d = [1, 0, 0, 0, 0x10, 0x32, 0x54, 0x76];
        let m = ModificationTimeMessage::parse(&d, Sizes::DEFAULT).unwrap();
        assert_eq!(m.seconds, 0x7654_3210);
        assert!(ModificationTimeMessage::parse(&d[..7], Sizes::DEFAULT).is_err());
        let mut d0 = d;
        d0[0] = 0;
        assert!(ModificationTimeMessage::parse(&d0, Sizes::DEFAULT).is_err());
    }

    #[test]
    fn old_style() {
        let m = ModificationTimeOldMessage::parse(b"20010203040506\0\0", Sizes::DEFAULT).unwrap();
        assert_eq!((m.year, m.month, m.day), (2001, 2, 3));
        assert_eq!((m.hour, m.minute, m.second), (4, 5, 6));
        assert_eq!(m.to_unix_seconds(), 981_173_106);
        let e = ModificationTimeOldMessage::parse(b"19700101000000", Sizes::DEFAULT).unwrap();
        assert_eq!(e.to_unix_seconds(), 0);
        assert!(ModificationTimeOldMessage::parse(b"2001020304050", Sizes::DEFAULT).is_err());
        assert!(ModificationTimeOldMessage::parse(b"2001130304050600", Sizes::DEFAULT).is_err());
        assert!(ModificationTimeOldMessage::parse(b"20x1020304050600", Sizes::DEFAULT).is_err());
    }
}
