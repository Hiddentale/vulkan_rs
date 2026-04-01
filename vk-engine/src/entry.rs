use std::sync::Arc;

use std::ffi::CStr;

use crate::error::{LoadError, VkResult, check, enumerate_two_call};
use crate::instance::Instance;
use crate::loader::Loader;
use crate::version::Version;
use crate::vk;
use vk::handles::Handle;

/// Entry point into the Vulkan API.
///
/// Loads the Vulkan shared library, resolves the bootstrap function pointers,
/// and provides access to entry-level commands (instance creation, version
/// query, layer/extension enumeration).
///
/// The `Entry` keeps the shared library alive via `Arc<dyn Loader>` for the
/// lifetime of all derived objects.
///
/// **Guide:** [Hello Triangle, Part 1](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html)
/// covers creating an `Entry` and bootstrapping the API.
///
/// # Examples
///
/// ```no_run
/// use vk_engine::{Entry, LibloadingLoader};
///
/// let loader = unsafe { LibloadingLoader::new() }.expect("Vulkan not found");
/// let entry = unsafe { Entry::new(loader) }.expect("entry creation failed");
///
/// let version = entry.version().expect("version query failed");
/// println!("Vulkan {version}");
/// ```
pub struct Entry {
    _loader: Arc<dyn Loader>,
    get_instance_proc_addr: vk::commands::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
    commands: vk::commands::EntryCommands,
}

impl Entry {
    /// Create a new `Entry` from the given loader.
    ///
    /// Resolves `vkGetInstanceProcAddr` from the library, then uses it to
    /// bootstrap `vkGetDeviceProcAddr` and all entry-level commands.
    ///
    /// # Safety
    ///
    /// The loader must return valid Vulkan function pointers. The loaded
    /// shared library must remain valid for the lifetime of this `Entry`
    /// and any objects created from it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vk_engine::{Entry, LibloadingLoader};
    ///
    /// let loader = unsafe { LibloadingLoader::new() }.expect("Vulkan not found");
    /// let entry = unsafe { Entry::new(loader) }.expect("entry creation failed");
    /// ```
    pub unsafe fn new(loader: impl Loader + 'static) -> Result<Self, LoadError> {
        let loader: Arc<dyn Loader> = Arc::new(loader);

        // SAFETY: loader returns a valid fn ptr or null; null is checked before transmute.
        let get_instance_proc_addr: vk::commands::PFN_vkGetInstanceProcAddr = unsafe {
            let ptr = loader.load(c"vkGetInstanceProcAddr");
            if ptr.is_null() {
                return Err(LoadError::MissingEntryPoint);
            }
            std::mem::transmute(ptr)
        };

        let get_instance_proc_addr_fn =
            get_instance_proc_addr.expect("vkGetInstanceProcAddr not loaded");
        let null_instance = vk::handles::Instance::null();

        // SAFETY: loader returns a valid fn ptr or null; null becomes None.
        let get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr =
            unsafe { std::mem::transmute(loader.load(c"vkGetDeviceProcAddr")) };

        // SAFETY: get_instance_proc_addr_fn is valid; null instance queries entry-level commands.
        let commands = unsafe {
            vk::commands::EntryCommands::load(|name| {
                std::mem::transmute(get_instance_proc_addr_fn(null_instance, name.as_ptr()))
            })
        };

        Ok(Self {
            _loader: loader,
            get_instance_proc_addr,
            get_device_proc_addr,
            commands,
        })
    }

    /// Returns the raw `vkGetInstanceProcAddr` function pointer.
    ///
    /// Needed by OpenXR's `XR_KHR_vulkan_enable2` which requires the
    /// application to provide this function pointer.
    pub fn get_instance_proc_addr(&self) -> vk::commands::PFN_vkGetInstanceProcAddr {
        self.get_instance_proc_addr
    }

    /// Returns the raw `vkGetDeviceProcAddr` function pointer.
    pub fn get_device_proc_addr(&self) -> vk::commands::PFN_vkGetDeviceProcAddr {
        self.get_device_proc_addr
    }

    /// Returns a reference to the loaded entry-level commands.
    pub(crate) fn commands(&self) -> &vk::commands::EntryCommands {
        &self.commands
    }

    /// Query the Vulkan instance version supported by the loader.
    ///
    /// Requires Vulkan 1.1+. On a 1.0-only system where
    /// `vkEnumerateInstanceVersion` is unavailable, returns `1.0.0`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # let entry = vk_engine::test_helpers::create_test_entry().unwrap();
    /// let version = entry.version().expect("version query failed");
    /// assert!(version.major >= 1);
    /// println!("Vulkan {version}");
    /// ```
    pub fn version(&self) -> VkResult<Version> {
        let fp = match self.commands.enumerate_instance_version {
            Some(fp) => fp,
            None => {
                return Ok(Version {
                    major: 1,
                    minor: 0,
                    patch: 0,
                });
            }
        };
        let mut raw = 0u32;
        // SAFETY: fp is vkEnumerateInstanceVersion; raw is a valid output pointer.
        check(unsafe { fp(&mut raw) })?;
        Ok(Version::from_raw(raw))
    }

    /// Create a Vulkan instance.
    ///
    /// # Safety
    ///
    /// `create_info` must be a valid, fully populated `InstanceCreateInfo`.
    /// The caller is responsible for calling `instance.destroy_instance`
    /// when done.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vk_engine::{Entry, LibloadingLoader};
    /// use vk_engine::vk::structs::*;
    ///
    /// let loader = unsafe { LibloadingLoader::new() }.expect("Vulkan not found");
    /// let entry = unsafe { Entry::new(loader) }.expect("entry creation failed");
    ///
    /// let app_info = ApplicationInfo::builder()
    ///     .api_version(1 << 22); // Vulkan 1.0
    /// let create_info = InstanceCreateInfo::builder()
    ///     .p_application_info(&*app_info);
    /// let instance = unsafe { entry.create_instance(&create_info, None) }
    ///     .expect("instance creation failed");
    /// // Use instance...
    /// unsafe { instance.destroy_instance(None) };
    /// ```
    pub unsafe fn create_instance(
        &self,
        create_info: &vk::structs::InstanceCreateInfo,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) -> VkResult<Instance> {
        // SAFETY: caller guarantees create_info is valid (this fn is unsafe).
        let raw = unsafe { self.create_instance_raw(create_info, allocator) }?;
        // SAFETY: raw is a freshly created valid instance handle.
        let instance = unsafe {
            Instance::load(
                raw,
                self.get_instance_proc_addr,
                self.get_device_proc_addr,
                Some(self._loader.clone()),
            )
        };
        Ok(instance)
    }

    /// Create a Vulkan instance and return the raw handle.
    ///
    /// Use this when you need the `VkInstance` handle without the wrapper,
    /// for example when passing it to OpenXR which manages the instance
    /// lifetime externally.
    ///
    /// # Safety
    ///
    /// `create_info` must be a valid, fully populated `InstanceCreateInfo`.
    /// The caller is responsible for destroying the instance with
    /// `vkDestroyInstance` when done.
    pub unsafe fn create_instance_raw(
        &self,
        create_info: &vk::structs::InstanceCreateInfo,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) -> VkResult<vk::handles::Instance> {
        let fp = self
            .commands
            .create_instance
            .expect("vkCreateInstance not loaded");
        let mut instance = vk::handles::Instance::null();
        // SAFETY: caller guarantees create_info is valid (this fn is unsafe).
        let result = unsafe {
            fp(
                create_info,
                allocator.map_or(std::ptr::null(), |a| a),
                &mut instance,
            )
        };
        check(result)?;
        Ok(instance)
    }

    /// Enumerate available instance layer properties.
    ///
    /// # Safety
    ///
    /// The Vulkan loader must be in a valid state.
    pub unsafe fn enumerate_instance_layer_properties(
        &self,
    ) -> VkResult<Vec<vk::structs::LayerProperties>> {
        let fp = self
            .commands
            .enumerate_instance_layer_properties
            .expect("vkEnumerateInstanceLayerProperties not loaded");
        // SAFETY: fp is vkEnumerateInstanceLayerProperties; two-call pattern handles allocation.
        enumerate_two_call(|count, data| unsafe { fp(count, data) })
    }

    /// Enumerate available instance extension properties.
    ///
    /// Pass `None` for `layer_name` to enumerate extensions provided by the
    /// loader and implicit layers. Pass a layer name to enumerate extensions
    /// provided by that layer.
    ///
    /// # Safety
    ///
    /// If `layer_name` is `Some`, it must name a layer that is present.
    pub unsafe fn enumerate_instance_extension_properties(
        &self,
        layer_name: Option<&CStr>,
    ) -> VkResult<Vec<vk::structs::ExtensionProperties>> {
        let fp = self
            .commands
            .enumerate_instance_extension_properties
            .expect("vkEnumerateInstanceExtensionProperties not loaded");
        let layer_ptr = layer_name.map_or(std::ptr::null(), |n| n.as_ptr());
        // SAFETY: fp is vkEnumerateInstanceExtensionProperties; layer_ptr is null or valid CStr.
        enumerate_two_call(|count, data| unsafe { fp(layer_ptr, count, data) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, c_char, c_void};

    /// Mock loader that returns null for everything,simulates missing library.
    struct NullLoader;

    unsafe impl Loader for NullLoader {
        unsafe fn load(&self, _name: &CStr) -> *const c_void {
            std::ptr::null()
        }
    }

    /// Mock loader that returns a fake `vkGetInstanceProcAddr` which itself
    /// returns null for everything. This lets us construct an Entry without
    /// a real Vulkan runtime.
    struct FakeEntryLoader;

    unsafe extern "system" fn mock_get_instance_proc_addr(
        _instance: vk::handles::Instance,
        _name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        None
    }

    unsafe impl Loader for FakeEntryLoader {
        unsafe fn load(&self, name: &CStr) -> *const c_void {
            if name == c"vkGetInstanceProcAddr" {
                mock_get_instance_proc_addr as *const c_void
            } else {
                std::ptr::null()
            }
        }
    }

    #[test]
    fn new_returns_missing_entry_point_when_loader_returns_null() {
        let result = unsafe { Entry::new(NullLoader) };
        match result {
            Err(LoadError::MissingEntryPoint) => {}
            Err(other) => panic!("expected MissingEntryPoint, got {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn new_succeeds_with_fake_loader() {
        let entry = unsafe { Entry::new(FakeEntryLoader) }.expect("should create Entry");
        assert!(entry.get_instance_proc_addr().is_some());
    }

    #[test]
    fn version_returns_1_0_when_enumerate_instance_version_is_none() {
        // FakeEntryLoader returns null for all commands, so
        // enumerate_instance_version will be None → 1.0 fallback.
        let entry = unsafe { Entry::new(FakeEntryLoader) }.expect("should create Entry");
        let version = entry.version().expect("version should succeed");
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
    }

    #[test]
    fn commands_returns_reference() {
        let entry = unsafe { Entry::new(FakeEntryLoader) }.expect("should create Entry");
        let _ = entry.commands();
    }

    #[test]
    fn get_instance_proc_addr_returns_some() {
        let entry = unsafe { Entry::new(FakeEntryLoader) }.expect("should create Entry");
        assert!(entry.get_instance_proc_addr().is_some());
    }

    #[test]
    fn get_device_proc_addr_returns_none_from_fake_loader() {
        // FakeEntryLoader returns null for vkGetDeviceProcAddr
        let entry = unsafe { Entry::new(FakeEntryLoader) }.expect("should create Entry");
        assert!(entry.get_device_proc_addr().is_none());
    }

    // -- Rich mock loader that returns fake entry-level function pointers ----

    /// Mock loader where `vkGetInstanceProcAddr` dispatches to fake
    /// implementations of entry-level commands.
    struct RichEntryLoader;

    unsafe extern "system" fn rich_get_instance_proc_addr(
        _instance: vk::handles::Instance,
        name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        let name = unsafe { CStr::from_ptr(name) };
        match name.to_bytes() {
            b"vkEnumerateInstanceVersion" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(*mut u32) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(mock_enumerate_instance_version)
            }),
            b"vkEnumerateInstanceLayerProperties" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *mut u32,
                        *mut vk::structs::LayerProperties,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(mock_enumerate_instance_layer_properties)
            }),
            b"vkEnumerateInstanceExtensionProperties" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *const c_char,
                        *mut u32,
                        *mut vk::structs::ExtensionProperties,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(mock_enumerate_instance_extension_properties)
            }),
            b"vkCreateInstance" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *const vk::structs::InstanceCreateInfo,
                        *const vk::structs::AllocationCallbacks,
                        *mut vk::handles::Instance,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(mock_create_instance)
            }),
            _ => None,
        }
    }

    unsafe extern "system" fn mock_enumerate_instance_version(
        p_api_version: *mut u32,
    ) -> vk::enums::Result {
        // Return Vulkan 1.3.290
        unsafe { *p_api_version = (1 << 22) | (3 << 12) | 290 };
        vk::enums::Result::SUCCESS
    }

    unsafe extern "system" fn mock_enumerate_instance_layer_properties(
        p_count: *mut u32,
        _p_properties: *mut vk::structs::LayerProperties,
    ) -> vk::enums::Result {
        unsafe { *p_count = 0 };
        vk::enums::Result::SUCCESS
    }

    unsafe extern "system" fn mock_enumerate_instance_extension_properties(
        _p_layer_name: *const c_char,
        p_count: *mut u32,
        _p_properties: *mut vk::structs::ExtensionProperties,
    ) -> vk::enums::Result {
        unsafe { *p_count = 0 };
        vk::enums::Result::SUCCESS
    }

    unsafe extern "system" fn mock_create_instance(
        _p_create_info: *const vk::structs::InstanceCreateInfo,
        _p_allocator: *const vk::structs::AllocationCallbacks,
        p_instance: *mut vk::handles::Instance,
    ) -> vk::enums::Result {
        // Write a non-null sentinel handle.
        unsafe { *p_instance = std::mem::transmute::<usize, vk::handles::Instance>(0x1234_usize) };
        vk::enums::Result::SUCCESS
    }

    unsafe impl Loader for RichEntryLoader {
        unsafe fn load(&self, name: &CStr) -> *const c_void {
            match name.to_bytes() {
                b"vkGetInstanceProcAddr" => rich_get_instance_proc_addr as *const c_void,
                b"vkGetDeviceProcAddr" => std::ptr::null(),
                _ => std::ptr::null(),
            }
        }
    }

    #[test]
    fn version_returns_parsed_version_when_fp_available() {
        let entry = unsafe { Entry::new(RichEntryLoader) }.expect("should create Entry");
        let version = entry.version().expect("version should succeed");
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 3);
        assert_eq!(version.patch, 290);
    }

    #[test]
    fn enumerate_layer_properties_with_mock() {
        let entry = unsafe { Entry::new(RichEntryLoader) }.expect("should create Entry");
        let layers =
            unsafe { entry.enumerate_instance_layer_properties() }.expect("should succeed");
        assert!(layers.is_empty());
    }

    #[test]
    fn enumerate_extension_properties_with_mock() {
        let entry = unsafe { Entry::new(RichEntryLoader) }.expect("should create Entry");
        let extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }.expect("should succeed");
        assert!(extensions.is_empty());
    }

    #[test]
    fn enumerate_extension_properties_with_layer_name() {
        let entry = unsafe { Entry::new(RichEntryLoader) }.expect("should create Entry");
        let extensions =
            unsafe { entry.enumerate_instance_extension_properties(Some(c"VK_LAYER_test")) }
                .expect("should succeed");
        assert!(extensions.is_empty());
    }

    #[test]
    fn create_instance_raw_with_mock() {
        let entry = unsafe { Entry::new(RichEntryLoader) }.expect("should create Entry");
        let create_info: vk::structs::InstanceCreateInfo = unsafe { std::mem::zeroed() };
        let raw = unsafe { entry.create_instance_raw(&create_info, None) }.expect("should succeed");
        assert!(!raw.is_null());
    }

    // -- Error-path mock loader -----------------------------------------------

    /// Mock loader where entry-level commands return Vulkan errors.
    struct FailingEntryLoader;

    unsafe extern "system" fn failing_get_instance_proc_addr(
        _instance: vk::handles::Instance,
        name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        let name = unsafe { CStr::from_ptr(name) };
        match name.to_bytes() {
            b"vkEnumerateInstanceVersion" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(*mut u32) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(failing_enumerate_instance_version)
            }),
            b"vkEnumerateInstanceLayerProperties" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *mut u32,
                        *mut vk::structs::LayerProperties,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(failing_enumerate_instance_layer_properties)
            }),
            b"vkEnumerateInstanceExtensionProperties" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *const c_char,
                        *mut u32,
                        *mut vk::structs::ExtensionProperties,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(failing_enumerate_instance_extension_properties)
            }),
            b"vkCreateInstance" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        *const vk::structs::InstanceCreateInfo,
                        *const vk::structs::AllocationCallbacks,
                        *mut vk::handles::Instance,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(failing_create_instance)
            }),
            _ => None,
        }
    }

    unsafe extern "system" fn failing_enumerate_instance_version(
        _p_api_version: *mut u32,
    ) -> vk::enums::Result {
        vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
    }

    unsafe extern "system" fn failing_enumerate_instance_layer_properties(
        _p_count: *mut u32,
        _p_properties: *mut vk::structs::LayerProperties,
    ) -> vk::enums::Result {
        vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
    }

    unsafe extern "system" fn failing_enumerate_instance_extension_properties(
        _p_layer_name: *const c_char,
        _p_count: *mut u32,
        _p_properties: *mut vk::structs::ExtensionProperties,
    ) -> vk::enums::Result {
        vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
    }

    unsafe extern "system" fn failing_create_instance(
        _p_create_info: *const vk::structs::InstanceCreateInfo,
        _p_allocator: *const vk::structs::AllocationCallbacks,
        _p_instance: *mut vk::handles::Instance,
    ) -> vk::enums::Result {
        vk::enums::Result::ERROR_INITIALIZATION_FAILED
    }

    unsafe impl Loader for FailingEntryLoader {
        unsafe fn load(&self, name: &CStr) -> *const c_void {
            match name.to_bytes() {
                b"vkGetInstanceProcAddr" => failing_get_instance_proc_addr as *const c_void,
                _ => std::ptr::null(),
            }
        }
    }

    #[test]
    fn version_propagates_error() {
        let entry = unsafe { Entry::new(FailingEntryLoader) }.expect("should create Entry");
        let result = entry.version();
        assert_eq!(
            result.unwrap_err(),
            vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
        );
    }

    #[test]
    fn create_instance_raw_propagates_error() {
        let entry = unsafe { Entry::new(FailingEntryLoader) }.expect("should create Entry");
        let create_info: vk::structs::InstanceCreateInfo = unsafe { std::mem::zeroed() };
        let result = unsafe { entry.create_instance_raw(&create_info, None) };
        assert_eq!(
            result.unwrap_err(),
            vk::enums::Result::ERROR_INITIALIZATION_FAILED
        );
    }

    #[test]
    fn enumerate_layer_properties_propagates_error() {
        let entry = unsafe { Entry::new(FailingEntryLoader) }.expect("should create Entry");
        let result = unsafe { entry.enumerate_instance_layer_properties() };
        assert_eq!(
            result.unwrap_err(),
            vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
        );
    }

    #[test]
    fn enumerate_extension_properties_propagates_error() {
        let entry = unsafe { Entry::new(FailingEntryLoader) }.expect("should create Entry");
        let result = unsafe { entry.enumerate_instance_extension_properties(None) };
        assert_eq!(
            result.unwrap_err(),
            vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY
        );
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn new_succeeds_with_real_loader() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let loader = crate::loader::LibloadingLoader::new().expect("failed to load Vulkan library");
        let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");
        assert!(entry.get_instance_proc_addr().is_some());
        assert!(entry.get_device_proc_addr().is_some());
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn version_returns_at_least_1_0() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let entry = create_entry();
        let version = entry.version().expect("failed to query version");
        assert!(version.major >= 1);
        println!("Vulkan {version}");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn enumerate_layer_properties_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let entry = create_entry();
        let layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("failed to enumerate layers");
        println!("found {} layers", layers.len());
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn enumerate_extension_properties_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let entry = create_entry();
        let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .expect("failed to enumerate extensions");
        assert!(!extensions.is_empty(), "expected at least one extension");
        println!("found {} extensions", extensions.len());
    }

    /// Helper to create an Entry for integration tests.
    fn create_entry() -> Entry {
        let loader = crate::loader::LibloadingLoader::new().expect("failed to load Vulkan library");
        unsafe { Entry::new(loader) }.expect("failed to create Entry")
    }
}
