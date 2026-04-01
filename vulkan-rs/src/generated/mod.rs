//! Generated wrapper methods for Entry, Instance, and Device.
//!
//! These methods are auto-generated from `vk.xml` by the `generator` crate.
//! Do not edit by hand,run `cargo run -p generator` to regenerate.
//!
//! Each method wraps a single Vulkan command, adding:
//! - Output-parameter returns (instead of out-pointer + `VkResult`)
//! - Two-call enumeration for array-returning commands
//! - Spec links, error codes, and safety documentation

mod device_wrappers;
mod entry_wrappers;
mod instance_wrappers;
