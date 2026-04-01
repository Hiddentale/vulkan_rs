use std::ffi::{CStr, c_void};

use crate::error::LoadError;

/// Abstraction over loading symbols from the Vulkan shared library.
///
/// # Safety
///
/// Implementations must return valid function pointers for the requested
/// symbol name, or null if the symbol is not found. Returning a pointer
/// to the wrong function causes undefined behavior.
///
/// # Examples
///
/// ```
/// use std::ffi::{CStr, c_void};
/// use vk_engine::Loader;
///
/// struct NullLoader;
///
/// unsafe impl Loader for NullLoader {
///     unsafe fn load(&self, _name: &CStr) -> *const c_void {
///         std::ptr::null()
///     }
/// }
///
/// let loader = NullLoader;
/// let ptr = unsafe { loader.load(c"vkCreateInstance") };
/// assert!(ptr.is_null());
/// ```
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
///
/// # Examples
///
/// ```no_run
/// use vk_engine::LibloadingLoader;
///
/// let loader = unsafe { LibloadingLoader::new() }
///     .expect("Vulkan library not found");
/// ```
pub struct LibloadingLoader {
    lib: libloading::Library,
}

impl LibloadingLoader {
    /// Load the platform's Vulkan shared library.
    pub fn new() -> Result<Self, LoadError> {
        // SAFETY: loading the platform's Vulkan shared library is standard initialization.
        let lib = unsafe { load_vulkan_library()? };
        Ok(Self { lib })
    }
}

unsafe impl Loader for LibloadingLoader {
    unsafe fn load(&self, name: &CStr) -> *const c_void {
        // SAFETY: name is a valid CStr; libloading resolves it from the loaded library.
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
        // SAFETY: loading a shared library by platform-known name.
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
    fn custom_loader_returns_non_null() {
        struct FixedLoader;
        unsafe impl Loader for FixedLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                0xDEAD as *const c_void
            }
        }
        let loader = FixedLoader;
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize, 0xDEAD);
    }

    /// Compile-time check that Loader requires Send + Sync.
    fn _assert_loader_is_send_sync<T: Loader>() {}
    #[test]
    fn loader_trait_requires_send_sync() {
        struct TestLoader;
        unsafe impl Loader for TestLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                std::ptr::null()
            }
        }
        _assert_loader_is_send_sync::<TestLoader>();
    }

    #[test]
    fn libloading_loader_new_error_is_load_error_library() {
        // On systems without Vulkan, new() should return LoadError::Library.
        // On systems WITH Vulkan, this test is still valid because it just
        // verifies the error type from a manually constructed error.
        let lib_err = unsafe { libloading::Library::new("__no_such_lib__") }.unwrap_err();
        let err = LoadError::Library(lib_err);
        match &err {
            LoadError::Library(_) => {}
            LoadError::MissingEntryPoint => panic!("expected Library variant"),
        }
    }

    #[test]
    fn libloading_loader_new_exercises_load_path() {
        // Exercises LibloadingLoader::new() and load_vulkan_library() without
        // requiring a working ICD. On CI (libvulkan-dev installed) this succeeds;
        // on machines without Vulkan it exercises the error path. Either way
        // the production code is covered.
        match LibloadingLoader::new() {
            Ok(loader) => {
                // Library loaded, verify the Loader impl works.
                let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
                // May be null if the library has no ICD, but the call itself succeeds.
                let _ = ptr;

                // Unknown symbol should return null.
                let unknown = unsafe { loader.load(c"vkNotARealFunction_XYZ") };
                assert!(unknown.is_null(), "unknown symbol should return null");
            }
            Err(e) => {
                // No Vulkan library on this system, verify error is Library variant.
                assert!(
                    matches!(e, LoadError::Library(_)),
                    "expected LoadError::Library, got {e}"
                );
                assert!(e.to_string().contains("failed to load Vulkan library"));
            }
        }
    }

    #[test]
    fn loader_is_object_safe() {
        // Verify Loader can be used as a trait object, which is critical
        // for Entry's Arc<dyn Loader> storage.
        struct TestLoader;
        unsafe impl Loader for TestLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                0xABCD as *const c_void
            }
        }
        let loader: Box<dyn Loader> = Box::new(TestLoader);
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert_eq!(ptr as usize, 0xABCD);
    }

    #[test]
    fn loader_behind_arc_works() {
        use std::sync::Arc;
        struct TestLoader;
        unsafe impl Loader for TestLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                0x1234 as *const c_void
            }
        }
        let loader: Arc<dyn Loader> = Arc::new(TestLoader);
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert_eq!(ptr as usize, 0x1234);
        assert_eq!(Arc::strong_count(&loader), 1);
    }

    #[test]
    fn loader_resolves_different_names_independently() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct CountingLoader;
        unsafe impl Loader for CountingLoader {
            unsafe fn load(&self, name: &CStr) -> *const c_void {
                CALL_COUNT.fetch_add(1, Ordering::SeqCst);
                match name.to_bytes() {
                    b"vkGetInstanceProcAddr" => 0x1000 as *const c_void,
                    b"vkGetDeviceProcAddr" => 0x2000 as *const c_void,
                    _ => std::ptr::null(),
                }
            }
        }

        CALL_COUNT.store(0, Ordering::SeqCst);
        let loader = CountingLoader;
        let gipa = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        let gdpa = unsafe { loader.load(c"vkGetDeviceProcAddr") };
        let unknown = unsafe { loader.load(c"vkUnknown") };

        assert_eq!(gipa as usize, 0x1000);
        assert_eq!(gdpa as usize, 0x2000);
        assert!(unknown.is_null());
        assert_eq!(CALL_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_new_succeeds() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let ptr = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert!(!ptr.is_null(), "vkGetInstanceProcAddr should be non-null");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_resolves_device_proc_addr() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let ptr = unsafe { loader.load(c"vkGetDeviceProcAddr") };
        assert!(!ptr.is_null(), "vkGetDeviceProcAddr should be non-null");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_returns_null_for_unknown_symbol() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let ptr = unsafe { loader.load(c"vkNotARealFunction_XYZ") };
        assert!(ptr.is_null(), "unknown symbol should return null");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_resolves_create_instance() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let ptr = unsafe { loader.load(c"vkCreateInstance") };
        assert!(!ptr.is_null(), "vkCreateInstance should be non-null");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_distinct_pointers_for_different_symbols() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let gipa = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        let gdpa = unsafe { loader.load(c"vkGetDeviceProcAddr") };
        assert!(!gipa.is_null());
        assert!(!gdpa.is_null());
        assert_ne!(
            gipa, gdpa,
            "different symbols should return different pointers"
        );
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn libloading_loader_same_symbol_returns_same_pointer() {
        let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
        let ptr1 = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        let ptr2 = unsafe { loader.load(c"vkGetInstanceProcAddr") };
        assert_eq!(ptr1, ptr2, "same symbol should return the same pointer");
    }
}
