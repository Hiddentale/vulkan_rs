//! Ergonomic Vulkan 1.2 wrapper with `from_raw_parts` support.

pub use vk_sys as vk;

mod device;
mod entry;
mod error;
mod instance;
mod loader;
mod version;

pub use device::Device;
pub use entry::Entry;
pub use error::{LoadError, VkResult};
pub use instance::Instance;
pub use loader::{LibloadingLoader, Loader};
pub use version::Version;
