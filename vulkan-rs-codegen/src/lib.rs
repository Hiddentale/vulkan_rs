//! Generator library,shared modules used by both the binary and tests.
//!
//! Internal modules are `#[doc(hidden)]`,they are not part of the public API
//! and may change without notice. They are exposed only for integration tests
//! and the generator binary.

#[doc(hidden)]
pub mod emit_aliases;
#[doc(hidden)]
pub mod emit_bitmasks;
#[doc(hidden)]
pub mod emit_builders;
#[doc(hidden)]
pub mod emit_commands;
#[doc(hidden)]
pub mod emit_constants;
#[doc(hidden)]
pub mod emit_enums;
#[doc(hidden)]
pub mod emit_handles;
#[doc(hidden)]
pub mod emit_layout_check;
#[doc(hidden)]
pub mod emit_structs;
#[doc(hidden)]
pub mod emit_wrappers;
#[doc(hidden)]
pub mod parse;
#[doc(hidden)]
pub mod resolve_types;
#[doc(hidden)]
pub mod stype;
#[doc(hidden)]
pub mod type_map;
#[doc(hidden)]
pub mod validate;
#[doc(hidden)]
pub mod wrapper_utils;
