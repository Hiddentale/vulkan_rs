#![warn(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

//! Ergonomic Vulkan wrapper built on generated FFI bindings.
//!
//! `vulkan-rs` provides safe-ish wrappers around the raw Vulkan API exposed by
//! [`vulkan-rs-sys`](vk). The raw types live in the re-exported [`vk`] module; this
//! crate adds ergonomic methods on [`Entry`], [`Instance`], and [`Device`] that
//! handle output parameters, two-call enumeration, and error checking.
//!
//! # Quick start
//!
//! ```no_run
//! use vulkan_rs::{Entry, LibloadingLoader};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let loader = unsafe { LibloadingLoader::new() }?;
//! let entry = unsafe { Entry::new(loader) }?;
//! // entry is now ready to create instances, enumerate layers, etc.
//! # Ok(())
//! # }
//! ```
//!
//! # Panics
//!
//! Wrapper methods on [`Instance`] and [`Device`] panic if the underlying
//! Vulkan function pointer was not loaded. This happens when calling a
//! command from an extension or Vulkan version that was not enabled at
//! instance or device creation time. The panic message names the missing
//! command (e.g., `"vkCmdDrawMeshTasksEXT not loaded"`).
//!
//! Core Vulkan 1.0 commands are always loaded and will not panic.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`vk`] | Re-export of `vulkan-rs-sys`,raw `#[repr(C)]` types, handles, enums |
//! | [`Entry`] | Vulkan entry point,loads the library, enumerates layers/extensions |
//! | [`Instance`] | Vulkan instance,physical device queries, instance-level commands |
//! | [`Device`] | Vulkan logical device,all device-level commands |
//! | [`bytecode`] | SPIR-V byte alignment helpers |
//! | [`Version`] | Decoded Vulkan version (major.minor.patch) from a packed `u32` |
//!
//! # Guide
//!
//! The companion [vulkan_rs Guide](https://hiddentale.github.io/vulkan_rs/)
//! covers Vulkan concepts in depth:
//!
//! | Topic | Guide chapter |
//! |-------|---------------|
//! | First steps | [Hello Triangle tutorial](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html) |
//! | Handles, lifetimes, parent-child | [The Vulkan Object Model](https://hiddentale.github.io/vulkan_rs/concepts/object-model.html) |
//! | Heaps, allocation, mapping | [Memory Management](https://hiddentale.github.io/vulkan_rs/concepts/memory.html) |
//! | Fences, semaphores, barriers | [Synchronization](https://hiddentale.github.io/vulkan_rs/concepts/synchronization.html) |
//! | Attachments, subpasses | [Render Passes & Framebuffers](https://hiddentale.github.io/vulkan_rs/concepts/render-passes.html) |
//! | Graphics & compute pipelines | [Pipelines](https://hiddentale.github.io/vulkan_rs/concepts/pipelines.html) |
//! | Layouts, pools, sets | [Descriptor Sets & Resource Binding](https://hiddentale.github.io/vulkan_rs/concepts/descriptors.html) |
//! | Recording & submission | [Command Buffers](https://hiddentale.github.io/vulkan_rs/concepts/command-buffers.html) |
//! | Extension struct chains | [The pNext Extension Chain](https://hiddentale.github.io/vulkan_rs/concepts/pnext.html) |
//! | Debug messenger, layers | [Validation Layers & Debugging](https://hiddentale.github.io/vulkan_rs/concepts/validation.html) |
//! | Safety model, two-crate design | [Design Decisions](https://hiddentale.github.io/vulkan_rs/architecture/design.html) |
//! | Error types, Result pattern | [Error Handling Philosophy](https://hiddentale.github.io/vulkan_rs/architecture/errors.html) |
//! | Porting from ash | [Migration Guide](https://hiddentale.github.io/vulkan_rs/how-to/migrate-from-ash.html) |
//! | C API to Rust mapping | [C-to-Rust Reference](https://hiddentale.github.io/vulkan_rs/how-to/c-to-rust.html) |
//!
//! # Feature flags
//!
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `surface` | yes | Enables [`required_extensions`] and [`SurfaceError`] for window surface creation via [`raw-window-handle`](https://docs.rs/raw-window-handle). Disable with `default-features = false` for headless use. |

pub use vulkan_rs_sys as vk;

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

// Static assertions: Entry, Instance, and Device must be Send + Sync.
// They are auto-derived (handle is usize, commands are Option<fn ptr>,
// loader is Option<Arc<dyn Loader>> where Loader: Send + Sync), but
// we pin this here so a future field change triggers a compile error.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn _check() {
        assert_send_sync::<Entry>();
        assert_send_sync::<Instance>();
        assert_send_sync::<Device>();
    }
};

/// Shared mutex for Vulkan runtime tests.
///
/// NVIDIA implicit layers (`VK_LAYER_NV_optimus`, `VK_LAYER_NV_present`)
/// are not thread-safe during concurrent `vkCreateInstance` calls. All
/// `#[ignore]` tests that create Vulkan instances must acquire this lock.
#[cfg(test)]
pub(crate) static VK_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
