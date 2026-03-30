//! Ergonomic Vulkan 1.2 wrapper with `from_raw_parts` support.

pub use vk_sys as vk;

pub mod bytecode;
mod device;
mod entry;
mod error;
mod instance;
mod loader;
#[cfg(feature = "surface")]
mod surface;
mod version;

pub use bytecode::{BytecodeError, cast_to_u32};
#[cfg(feature = "surface")]
pub use surface::required_extensions;
pub use device::Device;
pub use entry::Entry;
pub use error::{LoadError, VkResult};
pub use instance::Instance;
pub use loader::{LibloadingLoader, Loader};
pub use version::Version;

/// Shared mutex for Vulkan runtime tests.
///
/// NVIDIA implicit layers (`VK_LAYER_NV_optimus`, `VK_LAYER_NV_present`)
/// are not thread-safe during concurrent `vkCreateInstance` calls. All
/// `#[ignore]` tests that create Vulkan instances must acquire this lock.
#[cfg(test)]
pub(crate) static VK_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
