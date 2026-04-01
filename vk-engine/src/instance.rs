use std::sync::Arc;

use crate::device::Device;
use crate::error::{VkResult, check};
use crate::loader::Loader;
use crate::vk;
use vk::handles::Handle;

/// Wrapper around a `VkInstance` handle and its loaded command table.
///
/// Owns a `Box<InstanceCommands>` containing all instance-level function
/// pointers, loaded at construction via `vkGetInstanceProcAddr`. Also
/// stores `vkGetDeviceProcAddr` for later use when creating a `Device`.
///
/// Holds an optional reference to the Vulkan shared library so that
/// function pointers remain valid even if the originating `Entry` is
/// dropped. When created via `from_raw_parts`, the caller manages the
/// library lifetime and this field is `None`.
///
/// Does **not** implement `Drop`,the caller must explicitly call
/// `destroy_instance` when done. This avoids double-destroy bugs when
/// wrapping externally managed handles via `from_raw_parts`.
///
/// **Guide:** [The Vulkan Object Model](https://hiddentale.github.io/vulkan_rs/concepts/object-model.html)
/// covers handles, lifetimes, and parent-child relationships.
pub struct Instance {
    handle: vk::handles::Instance,
    commands: Box<vk::commands::InstanceCommands>,
    get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
    _loader: Option<Arc<dyn Loader>>,
}

impl Instance {
    /// Internal construction path. Called by `Entry::create_instance`.
    ///
    /// Loads all instance-level function pointers using the real instance
    /// handle, which gives instance-specific trampolines that skip a layer
    /// of dispatch compared to the global `vkGetInstanceProcAddr`.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid `VkInstance`.
    /// - `get_instance_proc_addr` must resolve instance-level commands for
    ///   this handle.
    /// - `get_device_proc_addr` must be the function used to load
    ///   device-level commands later.
    pub(crate) unsafe fn load(
        handle: vk::handles::Instance,
        get_instance_proc_addr: vk::commands::PFN_vkGetInstanceProcAddr,
        get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
        loader: Option<Arc<dyn Loader>>,
    ) -> Self {
        let get_instance_proc_addr_fn =
            get_instance_proc_addr.expect("vkGetInstanceProcAddr not loaded");
        // SAFETY: handle is valid per caller contract; transmute converts raw fn ptrs.
        let commands = Box::new(unsafe {
            vk::commands::InstanceCommands::load(|name| {
                std::mem::transmute(get_instance_proc_addr_fn(handle, name.as_ptr()))
            })
        });
        Self {
            handle,
            commands,
            get_device_proc_addr,
            _loader: loader,
        }
    }

    /// Wrap a raw handle created externally (OpenXR, middleware, testing).
    ///
    /// Resolves `vkGetDeviceProcAddr` from the instance automatically, so
    /// the caller only needs to provide `vkGetInstanceProcAddr`.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid `VkInstance` that was created externally.
    /// - `get_instance_proc_addr` must be the function used to load
    ///   instance-level commands for this handle.
    /// - The caller is responsible for the instance's lifetime,it must
    ///   outlive this wrapper and not be destroyed while in use.
    pub unsafe fn from_raw_parts(
        handle: vk::handles::Instance,
        get_instance_proc_addr: vk::commands::PFN_vkGetInstanceProcAddr,
    ) -> Self {
        let get_instance_proc_addr_fn =
            get_instance_proc_addr.expect("vkGetInstanceProcAddr not loaded");

        // SAFETY: resolving vkGetDeviceProcAddr from a valid instance handle.
        let get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr = unsafe {
            std::mem::transmute(get_instance_proc_addr_fn(
                handle,
                c"vkGetDeviceProcAddr".as_ptr(),
            ))
        };

        // SAFETY: forwards caller's safety guarantees to `load`.
        unsafe { Self::load(handle, get_instance_proc_addr, get_device_proc_addr, None) }
    }

    /// Returns the raw `VkInstance` handle.
    pub fn handle(&self) -> vk::handles::Instance {
        self.handle
    }

    /// Returns the loaded instance-level command table.
    ///
    /// Use this to call any of the ~90 instance-level commands directly,
    /// including those without hand-written ergonomic wrappers.
    pub fn commands(&self) -> &vk::commands::InstanceCommands {
        &self.commands
    }

    /// Create a logical device for the given physical device.
    ///
    /// # Safety
    ///
    /// - `physical_device` must be a valid handle obtained from this instance.
    /// - `create_info` must be a valid, fully populated `DeviceCreateInfo`.
    /// - The caller is responsible for calling `device.destroy_device` when done.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use vk_engine::vk::structs::*;
    /// # let (entry, instance) = vk_engine::test_helpers::create_test_instance().expect("test setup failed");
    /// let physical_devices = unsafe { instance.enumerate_physical_devices() }
    ///     .expect("no devices");
    /// let physical_device = physical_devices[0];
    ///
    /// let priorities = [1.0f32];
    /// let queue_info = DeviceQueueCreateInfo::builder()
    ///     .queue_family_index(0)
    ///     .queue_priorities(&priorities);
    /// let queue_infos = [*queue_info];
    /// let device_info = DeviceCreateInfo::builder()
    ///     .queue_create_infos(&queue_infos);
    /// let device = unsafe {
    ///     instance.create_device(physical_device, &device_info, None)
    /// }.expect("device creation failed");
    /// // Use device...
    /// unsafe { device.destroy_device(None) };
    /// # unsafe { instance.destroy_instance(None) };
    /// ```
    pub unsafe fn create_device(
        &self,
        physical_device: vk::handles::PhysicalDevice,
        create_info: &vk::structs::DeviceCreateInfo,
        allocator: Option<&vk::structs::AllocationCallbacks>,
    ) -> VkResult<Device> {
        let fp = self
            .commands
            .create_device
            .expect("vkCreateDevice not loaded");
        let mut raw = vk::handles::Device::null();
        // SAFETY: caller guarantees physical_device and create_info are valid.
        let result = unsafe {
            fp(
                physical_device,
                create_info,
                allocator.map_or(std::ptr::null(), |a| a),
                &mut raw,
            )
        };
        check(result)?;
        // SAFETY: raw is a freshly created valid device handle.
        let device = unsafe { Device::load(raw, self.get_device_proc_addr, self._loader.clone()) };
        Ok(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;
    use vk::handles::Handle;

    fn fake_handle() -> vk::handles::Instance {
        vk::handles::Instance::from_raw(0xDEAD)
    }

    /// Stub `vkGetInstanceProcAddr` that returns null for all lookups.
    unsafe extern "system" fn mock_get_instance_proc_addr(
        _instance: vk::handles::Instance,
        _name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        None
    }

    #[test]
    fn from_raw_parts_stores_handle() {
        let instance =
            unsafe { Instance::from_raw_parts(fake_handle(), Some(mock_get_instance_proc_addr)) };
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn handle_returns_value_from_construction() {
        let instance =
            unsafe { Instance::load(fake_handle(), Some(mock_get_instance_proc_addr), None, None) };
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn commands_returns_reference() {
        let instance =
            unsafe { Instance::load(fake_handle(), Some(mock_get_instance_proc_addr), None, None) };
        // Commands were loaded with a null-returning proc addr, so all
        // function pointers are None,but the reference is valid.
        let _ = instance.commands();
    }

    // -- Rich mock for testing create_device without Vulkan ------------------

    /// A `vkGetInstanceProcAddr` that returns fake fps for key commands.
    unsafe extern "system" fn rich_instance_proc_addr(
        _instance: vk::handles::Instance,
        name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        let name = unsafe { std::ffi::CStr::from_ptr(name) };
        match name.to_bytes() {
            b"vkGetDeviceProcAddr" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        vk::handles::Device,
                        *const c_char,
                    )
                        -> vk::structs::PFN_vkVoidFunction,
                    unsafe extern "system" fn(),
                >(mock_device_proc_addr)
            }),
            b"vkCreateDevice" => Some(unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn(
                        vk::handles::PhysicalDevice,
                        *const vk::structs::DeviceCreateInfo,
                        *const vk::structs::AllocationCallbacks,
                        *mut vk::handles::Device,
                    ) -> vk::enums::Result,
                    unsafe extern "system" fn(),
                >(mock_create_device)
            }),
            _ => None,
        }
    }

    unsafe extern "system" fn mock_device_proc_addr(
        _device: vk::handles::Device,
        _name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        None
    }

    unsafe extern "system" fn mock_create_device(
        _physical_device: vk::handles::PhysicalDevice,
        _p_create_info: *const vk::structs::DeviceCreateInfo,
        _p_allocator: *const vk::structs::AllocationCallbacks,
        p_device: *mut vk::handles::Device,
    ) -> vk::enums::Result {
        unsafe {
            *p_device = std::mem::transmute::<usize, vk::handles::Device>(0xBEEF_usize);
        }
        vk::enums::Result::SUCCESS
    }

    fn mock_instance() -> Instance {
        unsafe {
            Instance::load(
                fake_handle(),
                Some(rich_instance_proc_addr),
                Some(mock_device_proc_addr),
                None,
            )
        }
    }

    #[test]
    fn create_device_succeeds_with_mock() {
        let instance = mock_instance();
        let physical_device = vk::handles::PhysicalDevice::from_raw(0xCAFE);
        let create_info: vk::structs::DeviceCreateInfo = unsafe { std::mem::zeroed() };
        let device = unsafe { instance.create_device(physical_device, &create_info, None) }
            .expect("create_device should succeed");
        assert_eq!(device.handle().as_raw(), 0xBEEF);
    }

    #[test]
    fn create_device_with_allocator() {
        let instance = mock_instance();
        let physical_device = vk::handles::PhysicalDevice::from_raw(0xCAFE);
        let create_info: vk::structs::DeviceCreateInfo = unsafe { std::mem::zeroed() };
        let allocator: vk::structs::AllocationCallbacks = unsafe { std::mem::zeroed() };
        let device =
            unsafe { instance.create_device(physical_device, &create_info, Some(&allocator)) }
                .expect("create_device should succeed");
        assert_eq!(device.handle().as_raw(), 0xBEEF);
    }

    #[test]
    fn from_raw_parts_resolves_device_proc_addr() {
        // rich_instance_proc_addr returns a non-null vkGetDeviceProcAddr,
        // so from_raw_parts should store it.
        let instance =
            unsafe { Instance::from_raw_parts(fake_handle(), Some(rich_instance_proc_addr)) };
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
        // Verify create_device works, proving get_device_proc_addr was resolved.
        let physical_device = vk::handles::PhysicalDevice::from_raw(0xCAFE);
        let create_info: vk::structs::DeviceCreateInfo = unsafe { std::mem::zeroed() };
        let device = unsafe { instance.create_device(physical_device, &create_info, None) }
            .expect("create_device should succeed via from_raw_parts path");
        assert_eq!(device.handle().as_raw(), 0xBEEF);
    }

    #[test]
    fn load_with_loader_reference() {
        use std::ffi::{CStr, c_void};
        struct DummyLoader;
        unsafe impl Loader for DummyLoader {
            unsafe fn load(&self, _name: &CStr) -> *const c_void {
                std::ptr::null()
            }
        }
        let loader: Arc<dyn Loader> = Arc::new(DummyLoader);
        let instance = unsafe {
            Instance::load(
                fake_handle(),
                Some(mock_get_instance_proc_addr),
                None,
                Some(loader.clone()),
            )
        };
        // The loader Arc should have 2 strong refs: our local + instance.
        assert_eq!(Arc::strong_count(&loader), 2);
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn enumerate_physical_devices_returns_at_least_one() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let instance = create_real_instance();
        let devices = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate_physical_devices failed");
        assert!(!devices.is_empty(), "expected at least one physical device");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn get_physical_device_properties_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let instance = create_real_instance();
        let devices = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate_physical_devices failed");
        let props = unsafe { instance.get_physical_device_properties(devices[0]) };
        let name_bytes: Vec<u8> = props
            .device_name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        let name = String::from_utf8_lossy(&name_bytes);
        println!("GPU: {name}");
        assert!(!name.is_empty());
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn get_physical_device_queue_family_properties_returns_at_least_one() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let instance = create_real_instance();
        let devices = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate_physical_devices failed");
        let families = unsafe { instance.get_physical_device_queue_family_properties(devices[0]) };
        assert!(!families.is_empty(), "expected at least one queue family");
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn get_physical_device_memory_properties_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let instance = create_real_instance();
        let devices = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate_physical_devices failed");
        let mem_props = unsafe { instance.get_physical_device_memory_properties(devices[0]) };
        assert!(
            mem_props.memory_type_count > 0,
            "expected at least one memory type"
        );
        assert!(
            mem_props.memory_heap_count > 0,
            "expected at least one memory heap"
        );
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn get_physical_device_features_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let instance = create_real_instance();
        let devices = unsafe { instance.enumerate_physical_devices() }
            .expect("enumerate_physical_devices failed");
        // Just verify the call completes without crashing; feature
        // availability is driver-dependent.
        let _features = unsafe { instance.get_physical_device_features(devices[0]) };
    }

    fn create_real_instance() -> Instance {
        use crate::entry::Entry;
        use crate::loader::LibloadingLoader;

        let loader = LibloadingLoader::new().expect("failed to load Vulkan");
        let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");

        // VK_MAKE_API_VERSION(0, 1, 0, 0)
        let api_version_1_0 = 1u32 << 22;

        let app_info = vk::structs::ApplicationInfo {
            s_type: vk::enums::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: std::ptr::null(),
            application_version: 0,
            p_engine_name: std::ptr::null(),
            engine_version: 0,
            api_version: api_version_1_0,
        };
        let create_info = vk::structs::InstanceCreateInfo {
            s_type: vk::enums::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::bitmasks::InstanceCreateFlagBits::empty(),
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };

        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance")
    }
}
