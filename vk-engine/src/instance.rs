use crate::vk;

/// Wrapper around a `VkInstance` handle and its loaded command table.
///
/// Owns a `Box<InstanceCommands>` containing all instance-level function
/// pointers, loaded at construction via `vkGetInstanceProcAddr`. Also
/// stores `vkGetDeviceProcAddr` for later use when creating a `Device`.
///
/// Does **not** implement `Drop` — the caller must explicitly call
/// `destroy_instance` when done. This avoids double-destroy bugs when
/// wrapping externally managed handles via `from_raw_parts`.
pub struct Instance {
    handle: vk::handles::Instance,
    commands: Box<vk::commands::InstanceCommands>,
    #[allow(dead_code)] // used by create_device in subtask 1d
    get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr,
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
    ) -> Self {
        let get_instance_proc_addr_fn = get_instance_proc_addr.unwrap();
        let commands = Box::new(unsafe {
            vk::commands::InstanceCommands::load(|name| {
                std::mem::transmute(get_instance_proc_addr_fn(handle, name.as_ptr()))
            })
        });
        Self {
            handle,
            commands,
            get_device_proc_addr,
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
    /// - The caller is responsible for the instance's lifetime — it must
    ///   outlive this wrapper and not be destroyed while in use.
    pub unsafe fn from_raw_parts(
        handle: vk::handles::Instance,
        get_instance_proc_addr: vk::commands::PFN_vkGetInstanceProcAddr,
    ) -> Self {
        let get_instance_proc_addr_fn = get_instance_proc_addr.unwrap();

        let get_device_proc_addr: vk::commands::PFN_vkGetDeviceProcAddr = unsafe {
            std::mem::transmute(get_instance_proc_addr_fn(
                handle,
                c"vkGetDeviceProcAddr".as_ptr(),
            ))
        };

        unsafe { Self::load(handle, get_instance_proc_addr, get_device_proc_addr) }
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
        let instance = unsafe {
            Instance::from_raw_parts(fake_handle(), Some(mock_get_instance_proc_addr))
        };
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn handle_returns_value_from_construction() {
        let instance = unsafe {
            Instance::load(
                fake_handle(),
                Some(mock_get_instance_proc_addr),
                None,
            )
        };
        assert_eq!(instance.handle().as_raw(), fake_handle().as_raw());
    }

    #[test]
    fn commands_returns_reference() {
        let instance = unsafe {
            Instance::load(
                fake_handle(),
                Some(mock_get_instance_proc_addr),
                None,
            )
        };
        // Commands were loaded with a null-returning proc addr, so all
        // function pointers are None — but the reference is valid.
        let _ = instance.commands();
    }
}
