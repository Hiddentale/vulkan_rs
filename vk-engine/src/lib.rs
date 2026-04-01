//! Ergonomic Vulkan wrapper built on generated FFI bindings.
//!
//! `vk-engine` provides safe-ish wrappers around the raw Vulkan API exposed by
//! [`vk-sys`](vk). The raw types live in the re-exported [`vk`] module; this
//! crate adds ergonomic methods on [`Entry`], [`Instance`], and [`Device`] that
//! handle output parameters, two-call enumeration, and error checking.
//!
//! # Quick start
//!
//! ```no_run
//! use vk_engine::{Entry, LibloadingLoader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let loader = unsafe { LibloadingLoader::new() }?;
//! let entry = unsafe { Entry::new(loader) }?;
//! // entry is now ready to create instances, enumerate layers, etc.
//! # Ok(())
//! # }
//! ```
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`vk`] | Re-export of `vk-sys`,raw `#[repr(C)]` types, handles, enums |
//! | [`Entry`] | Vulkan entry point,loads the library, enumerates layers/extensions |
//! | [`Instance`] | Vulkan instance,physical device queries, instance-level commands |
//! | [`Device`] | Vulkan logical device,all device-level commands |
//! | [`bytecode`] | SPIR-V byte alignment helpers |
//! | [`Version`] | Decoded Vulkan version (major.minor.patch) from a packed `u32` |
//!
//! # Feature flags
//!
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `surface` | yes | Enables [`required_extensions`] and [`SurfaceError`] for window surface creation via [`raw-window-handle`](https://docs.rs/raw-window-handle). Disable with `default-features = false` for headless use. |

pub use vk_sys as vk;

pub mod bytecode;
mod device;
mod entry;
mod error;
mod generated;
mod instance;
mod loader;
#[cfg(feature = "surface")]
mod surface;
#[doc(hidden)]
pub mod test_helpers;
mod version;

pub use bytecode::{BytecodeError, cast_to_u32};
pub use device::Device;
pub use entry::Entry;
pub use error::{LoadError, VkResult};
pub use instance::Instance;
pub use loader::{LibloadingLoader, Loader};
#[cfg(feature = "surface")]
pub use surface::{SurfaceError, required_extensions};
pub use version::Version;

/// Shared mutex for Vulkan runtime tests.
///
/// NVIDIA implicit layers (`VK_LAYER_NV_optimus`, `VK_LAYER_NV_present`)
/// are not thread-safe during concurrent `vkCreateInstance` calls. All
/// `#[ignore]` tests that create Vulkan instances must acquire this lock.
#[cfg(test)]
pub(crate) static VK_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
