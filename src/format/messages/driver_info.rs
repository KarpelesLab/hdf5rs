//! Driver Info message (type 0x0014, spec IV.A.2.u).

use crate::error::{Error, Result};
use crate::format::{ByteReader, Sizes};

/// File driver information (superblock extension only). Parsed for
/// completeness; the reader does not act on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverInfoMessage {
    /// Eight-byte ASCII driver identification.
    pub driver_id: [u8; 8],
    /// Driver-defined payload.
    pub info: Vec<u8>,
}

impl DriverInfoMessage {
    /// Decode the message body.
    pub fn parse(data: &[u8], _sizes: Sizes) -> Result<Self> {
        let mut r = ByteReader::new(data);
        let version = r.read_u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "driver info message version {version}"
            )));
        }
        let driver_id = r.read_array::<8>()?;
        let size = r.read_u16()? as usize;
        let info = r.read_bytes(size)?.to_vec();
        Ok(DriverInfoMessage { driver_id, info })
    }

    /// The driver identification as text (lossy).
    pub fn driver_id_str(&self) -> String {
        String::from_utf8_lossy(&self.driver_id).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_util::W;

    #[test]
    fn parse() {
        let mut w = W::new();
        w.u8(0).bytes(b"NCSAfami").u16(8).u64(1 << 31);
        let m = DriverInfoMessage::parse(&w.buf, Sizes::DEFAULT).unwrap();
        assert_eq!(m.driver_id_str(), "NCSAfami");
        assert_eq!(m.info.len(), 8);
        assert!(DriverInfoMessage::parse(&w.buf[..w.len() - 1], Sizes::DEFAULT).is_err());
    }
}
