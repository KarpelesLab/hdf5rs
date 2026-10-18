//! Object header message bodies (spec: "IV.A.2. Disk Format: Level 2A2 -
//! Data Object Header Messages").
//!
//! Each submodule decodes the body of one message type from a byte slice,
//! honouring the file's [`Sizes`](crate::format::Sizes).

pub mod attribute;
pub mod attribute_info;
pub mod dataspace;
pub mod datatype;
pub mod external_files;
pub mod fill_value;
pub mod filter_pipeline;
pub mod layout;
