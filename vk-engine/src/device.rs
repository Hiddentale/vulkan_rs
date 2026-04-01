use std::sync::Arc;

use crate::loader::Loader;
use crate::vk;

/// Wrapper around a `VkDevice` handle and its loaded command table.
///
/// Owns a `Box<DeviceCommands>` containing all device-level function
/// pointers, loaded at construction via `vkGetDeviceProcAddr`. Using the
/// real device handle gives the ICD's direct function pointers, bypassing
/// the loader trampoline,this is the fastest dispatch path in Vulkan.
///
/// Holds an optional reference to the Vulkan shared library so that
/// function pointers remain valid even if the originating `Entry` is
/// dropped. When created via `from_raw_parts`, the caller manages the
/// library lifetime and this field is `None`.
///
/// Does **not** implement `Drop`,the caller must explicitly call
/// `destroy_device` when done. This avoids double-destroy bugs when
/// wrapping externally managed handles via `from_raw_parts`.
pub struct Device {
    handle: vk::handles::Device,
    commands: Box<vk::commands::DeviceCommands>,
    _loader: Option<Arc<dyn Loader>>,
}

impl Device {
    /// Internal construction path. Called by `Instance::create_device`.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid `VkDevice`.
    /// - `get_device_proc_addr` must resolve device-level commands for
    ///   this handle.
    pub(crate) unsafe fn load(
        handle: vk::handles::Device,
        get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
        loader: Option<Arc<dyn Loader>>,
    ) -> Self {
        let get_device_proc_addr_fn = get_device_proc_addr.expect("vkGetDeviceProcAddr not loaded");
        // SAFETY: handle is valid per caller contract; transmute converts raw fn ptrs.
        let commands = Box::new(unsafe {
            vk::commands::DeviceCommands::load(|name| {
                std::mem::transmute(get_device_proc_addr_fn(handle, name.as_ptr()))
            })
        });
        Self {
            handle,
            commands,
            _loader: loader,
        }
    }

    /// Wrap a raw handle created externally (OpenXR, middleware, testing).
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid `VkDevice`.
    /// - `get_device_proc_addr` must resolve commands for this device.
    /// - The caller owns the device lifetime.
    pub unsafe fn from_raw_parts(
        handle: vk::handles::Device,
        get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
    ) -> Self {
        // SAFETY: forwards caller's safety guarantees to `load`.
        unsafe { Self::load(handle, get_device_proc_addr, None) }
    }

    /// Returns the raw `VkDevice` handle.
    pub fn handle(&self) -> vk::handles::Device {
        self.handle
    }

    /// Returns the loaded device-level command table.
    ///
    /// Use this to call any of the ~200 device-level commands directly,
    /// including those without hand-written ergonomic wrappers.
    pub fn commands(&self) -> &vk::commands::DeviceCommands {
        &self.commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_char;
    use vk::handles::Handle;

    fn fake_handle() -> vk::handles::Device {
        vk::handles::Device::from_raw(0xBEEF)
    }

    /// Stub `vkGetDeviceProcAddr` that returns null for all lookups.
    unsafe extern "system" fn mock_get_device_proc_addr(
        _device: vk::handles::Device,
        _name: *const c_char,
    ) -> vk::structs::PFN_vkVoidFunction {
        None
    }

    #[test]
    fn from_raw_parts_stores_handle() {
        let device =
            unsafe { Device::from_raw_parts(fake_handle(), Some(mock_get_device_proc_addr)) };
        assert_eq!(device.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn handle_returns_value_from_construction() {
        let device = unsafe { Device::load(fake_handle(), Some(mock_get_device_proc_addr), None) };
        assert_eq!(device.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn commands_returns_reference() {
        let device = unsafe { Device::load(fake_handle(), Some(mock_get_device_proc_addr), None) };
        // Commands were loaded with a null-returning proc addr, so all
        // function pointers are None,but the reference is valid.
        let _ = device.commands();
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn device_wait_idle_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let (instance, device) = create_real_device();
        unsafe { device.device_wait_idle() }.expect("device_wait_idle failed");
        unsafe { device.destroy_device(None) };
        unsafe { instance.destroy_instance(None) };
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn get_device_queue_returns_non_null_queue() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let (instance, device) = create_real_device();
        let queue = unsafe { device.get_device_queue(0, 0) };
        assert!(!queue.is_null(), "expected non-null queue handle");
        unsafe { device.destroy_device(None) };
        unsafe { instance.destroy_instance(None) };
    }

    #[test]
    #[ignore] // requires Vulkan runtime
    fn queue_wait_idle_succeeds() {
        let _vk = crate::VK_TEST_MUTEX.lock().expect("VK_TEST_MUTEX poisoned");
        let (instance, device) = create_real_device();
        let queue = unsafe { device.get_device_queue(0, 0) };
        unsafe { device.queue_wait_idle(queue) }.expect("queue_wait_idle failed");
        unsafe { device.destroy_device(None) };
        unsafe { instance.destroy_instance(None) };
    }

    fn create_real_device() -> (crate::instance::Instance, Device) {
        use crate::entry::Entry;
        use crate::loader::LibloadingLoader;

        let loader = LibloadingLoader::new().expect("failed to load Vulkan");
        let entry = unsafe { Entry::new(loader) }.expect("failed to create Entry");

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
        let instance_create_info = vk::structs::InstanceCreateInfo {
            s_type: vk::enums::StructureType::INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::bitmasks::InstanceCreateFlagBits::empty(),
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .expect("failed to create instance");

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("failed to enumerate physical devices");
        let physical_device = physical_devices[0];

        let queue_priority = 1.0f32;
        let queue_create_info = vk::structs::DeviceQueueCreateInfo {
            s_type: vk::enums::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::bitmasks::DeviceQueueCreateFlagBits::empty(),
            queue_family_index: 0,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
        };
        let device_create_info = vk::structs::DeviceCreateInfo {
            s_type: vk::enums::StructureType::DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
        };
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .expect("failed to create device");

        (instance, device)
    }
}
