//! Raw Vulkan FFI types generated from `vk.xml`.
//!
//! Do not edit by hand,regenerate with the `generator` crate.
//!
//! Every type carries a spec link, and structs include metadata from
//! vk.xml: extension provenance, pNext chain relationships, member
//! annotations (optional, length-of, thread safety).

#![no_std]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod handles;
pub mod enums;
pub mod bitmasks;
pub mod constants;
pub mod structs;
pub mod builders;
pub mod commands;
