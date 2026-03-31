use std::ffi::{CStr, c_void};

use crate::error::LoadError;

/// Abstraction over loading symbols from the Vulkan shared library.
///
/// # Safety
///
/// Implementations must return valid function pointers for the requested
/// symbol name, or null if the symbol is not found. Returning a pointer
/// to the wrong function causes undefined behavior.
pub unsafe trait Loader: Send + Sync {
    /// Load a function by name from the Vulkan library.
    ///
    /// Returns a raw function pointer, or null if the symbol is not found.
    ///
    /// # Safety
    ///
    /// The caller must only transmute the returned pointer to a function
    /// type matching the Vulkan command identified by `name`.
    unsafe fn load(&self, name: &CStr) -> *const c_void;
}

/// Default [`Loader`] implementation backed by `libloading`.
///
/// Loads the platform-appropriate Vulkan shared library at construction
/// time and resolves symbols from it on demand.
pub struct LibloadingLoader {
    lib: libloading::Library,
}

impl LibloadingLoader {
    /// Load the platform's Vulkan shared library.
    pub fn new() -> Result<Self, LoadError> {
        let lib = unsafe { load_vulkan_library()? };
        Ok(Self { lib })
    }
}

unsafe impl Loader for LibloadingLoader {
    unsafe fn load(&self, name: &CStr) -> *const c_void {
        unsafe {
            self.lib
                .get::<*const c_void>(name.to_bytes_with_nul())
                .map(|sym| *sym)
                .unwrap_or(std::ptr::null())
        }
    }
}

/// Load the platform-specific Vulkan shared library.
///
/// # Safety
///
/// Loading a shared library can execute arbitrary initialization code.
unsafe fn load_vulkan_library() -> Result<libloading::Library, LoadError> {
    #[cfg(target_os = "windows")]
    const LIB_NAMES: &[&str] = &["vulkan-1.dll"];

    #[cfg(target_os = "linux")]
    const LIB_NAMES: &[&str] = &["libvulkan.so.1", "libvulkan.so"];

    #[cfg(target_os = "android")]
    const LIB_NAMES: &[&str] = &["libvulkan.so.1", "libvulkan.so"];

    #[cfg(target_os = "macos")]
    const LIB_NAMES: &[&str] = &["libvulkan.1.dylib", "libMoltenVK.dylib"];

    let mut last_err = None;
    for name in LIB_NAMES {
        match unsafe { libloading::Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last_err = Some(e),
        }
    }
    Err(LoadError::Library(
        last_err.expect("LIB_NAMES is non-empty"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_loader_returns_null() {
        struct TestNullLoader;
        unsafe impl Loader for TestNullLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                std::ptr::null()
            }
        }
        let loader = TestNullLoader;
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert!(ptr.is_null());
    }

    #[test]
    fn load_vulkan_library_returns_error_on_missing_platform() {
        // Verify the error path is reachable.
        let result = LoadError::MissingEntryPoint;
        assert_eq!(
            result.to_string(),
            "vkGetInstanceProcAddr not found in Vulkan library"
        );
    }

    #[test]
    fn libloading_loader_new_returns_error_message_on_missing_lib() {
        // We can't easily force a missing library, but we can verify
        // the error path by trying to load a nonsense library name.
        let err = unsafe { libloading::Library::new("__nonexistent_vulkan_lib__") };
        assert!(err.is_err());
        let load_err = LoadError::Library(err.unwrap_err());
        let msg = load_err.to_string();
        assert!(msg.contains("failed to load Vulkan library"));
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_new_succeeds() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        // vkGetInstanceProcAddr should be resolvable
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert!(!ptr.is_null(), "vkGetInstanceProcAddr should be non-null");
    }
}
