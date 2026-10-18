//! Error type shared by the whole crate.

use std::borrow::Cow;
use std::fmt;

/// Convenience alias for `Result<T, hdf5rs::Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors returned by this crate.
///
/// Malformed input must always surface as an error, never as a panic.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An error from the underlying storage.
    Io(std::io::Error),
    /// A structure ended before all of its fields could be read.
    UnexpectedEof,
    /// The file violates the HDF5 file format specification.
    InvalidFormat(Cow<'static, str>),
    /// The file uses a valid feature this crate does not support (yet).
    Unsupported(Cow<'static, str>),
    /// A metadata checksum did not match.
    ChecksumMismatch {
        /// What structure was being verified.
        what: &'static str,
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum computed over the data.
        actual: u32,
    },
    /// A size, count or depth exceeded a sanity limit.
    LimitExceeded(&'static str),
    /// A named object, link or attribute does not exist.
    NotFound(String),
}

impl Error {
    /// Shorthand for [`Error::InvalidFormat`].
    pub fn invalid(msg: impl Into<Cow<'static, str>>) -> Self {
        Error::InvalidFormat(msg.into())
    }

    /// Shorthand for [`Error::Unsupported`].
    pub fn unsupported(msg: impl Into<Cow<'static, str>>) -> Self {
        Error::Unsupported(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::UnexpectedEof => f.write_str("unexpected end of data"),
            Error::InvalidFormat(m) => write!(f, "invalid HDF5 data: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported HDF5 feature: {m}"),
            Error::ChecksumMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch in {what}: stored {expected:#010x}, computed {actual:#010x}"
            ),
            Error::LimitExceeded(what) => write!(f, "limit exceeded: {what}"),
            Error::NotFound(name) => write!(f, "not found: {name}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::UnexpectedEof
        } else {
            Error::Io(e)
        }
    }
}
