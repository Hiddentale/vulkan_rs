//! Ergonomic Vulkan 1.2 wrapper with `from_raw_parts` support.

pub use vk_sys as vk;

mod entry;
mod error;
mod loader;
mod version;

pub use entry::Entry;
pub use error::{LoadError, VkResult};
pub use loader::{LibloadingLoader, Loader};
pub use version::Version;
