#![no_main]

use libfuzzer_sys::fuzz_target;
use std::ffi::{CStr, c_void};
use vk_engine::Loader;

/// A loader whose `load` return values are driven by fuzz input.
///
/// The fuzzer controls:
/// - byte 0: mode for vkGetInstanceProcAddr (0 = null, 1+ = sentinel)
/// - byte 1: mode for vkGetDeviceProcAddr (0 = null, 1+ = sentinel)
/// - byte 2: mode for all other symbols  (0 = null, 1+ = sentinel)
///
/// When mode is non-zero, we return a pointer to a real extern "system" fn
/// that does nothing, simulating a driver that provides the symbol. This
/// exercises the transmute + Option wrapping paths in Entry::new without
/// invoking actual Vulkan commands afterward.
struct FuzzLoader {
    gipa_mode: u8,
    gdpa_mode: u8,
    other_mode: u8,
}

/// Minimal valid vkGetInstanceProcAddr: returns null for all lookups.
unsafe extern "system" fn stub_get_instance_proc_addr(
    _instance: vk_sys::handles::Instance,
    _name: *const std::ffi::c_char,
) -> vk_sys::structs::PFN_vkVoidFunction {
    None
}

/// Minimal valid vkGetDeviceProcAddr: returns null for all lookups.
unsafe extern "system" fn stub_get_device_proc_addr(
    _device: vk_sys::handles::Device,
    _name: *const std::ffi::c_char,
) -> vk_sys::structs::PFN_vkVoidFunction {
    None
}

unsafe impl Loader for FuzzLoader {
    unsafe fn load(&self, name: &CStr) -> *const c_void {
        match name.to_bytes() {
            b"vkGetInstanceProcAddr" => {
                if self.gipa_mode == 0 {
                    std::ptr::null()
                } else {
                    stub_get_instance_proc_addr as *const c_void
                }
            }
            b"vkGetDeviceProcAddr" => {
                if self.gdpa_mode == 0 {
                    std::ptr::null()
                } else {
                    stub_get_device_proc_addr as *const c_void
                }
            }
            _ => {
                if self.other_mode == 0 {
                    std::ptr::null()
                } else {
                    // Return a non-null but unusable pointer for unknown symbols.
                    // Entry::new only transmutes gipa/gdpa, other symbols go through
                    // EntryCommands::load which wraps them in Option (None if null).
                    // Non-null here means the command appears "loaded" but would
                    // crash if called, which is fine since we never call them.
                    stub_get_instance_proc_addr as *const c_void
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let loader = FuzzLoader {
        gipa_mode: data[0],
        gdpa_mode: data[1],
        other_mode: data[2],
    };

    match unsafe { vk_engine::Entry::new(loader) } {
        Ok(entry) => {
            // Entry created successfully. gipa must have been non-null.
            assert!(entry.get_instance_proc_addr().is_some());

            // Exercise the version query path (uses EntryCommands).
            // This should not panic regardless of command availability.
            let _ = entry.version();
        }
        Err(vk_engine::LoadError::MissingEntryPoint) => {
            // gipa was null, which is correct when gipa_mode == 0.
            assert_eq!(data[0], 0);
        }
        Err(vk_engine::LoadError::Library(_)) => {
            // Should never happen with our in-memory loader.
            panic!("unexpected LoadError::Library from FuzzLoader");
        }
    }
});
