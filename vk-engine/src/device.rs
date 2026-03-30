use crate::vk;

/// Wrapper around a `VkDevice` handle and its loaded command table.
///
/// Owns a `Box<DeviceCommands>` containing all device-level function
/// pointers, loaded at construction via `vkGetDeviceProcAddr`. Using the
/// real device handle gives the ICD's direct function pointers, bypassing
/// the loader trampoline — this is the fastest dispatch path in Vulkan.
///
/// Does **not** implement `Drop` — the caller must explicitly call
/// `destroy_device` when done. This avoids double-destroy bugs when
/// wrapping externally managed handles via `from_raw_parts`.
pub struct Device {
    handle: vk::handles::Device,
    commands: Box<vk::commands::DeviceCommands>,
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
    ) -> Self {
        let get_device_proc_addr_fn = get_device_proc_addr.unwrap();
        let commands = Box::new(unsafe {
            vk::commands::DeviceCommands::load(|name| {
                std::mem::transmute(get_device_proc_addr_fn(handle, name.as_ptr()))
            })
        });
        Self { handle, commands }
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
        unsafe { Self::load(handle, get_device_proc_addr) }
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
        let device = unsafe {
            Device::from_raw_parts(fake_handle(), Some(mock_get_device_proc_addr))
        };
        assert_eq!(device.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn handle_returns_value_from_construction() {
        let device = unsafe {
            Device::load(fake_handle(), Some(mock_get_device_proc_addr))
        };
        assert_eq!(device.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn commands_returns_reference() {
        let device = unsafe {
            Device::load(fake_handle(), Some(mock_get_device_proc_addr))
        };
        // Commands were loaded with a null-returning proc addr, so all
        // function pointers are None — but the reference is valid.
        let _ = device.commands();
    }
}
