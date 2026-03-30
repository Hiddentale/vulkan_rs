//! Struct layout tests — verify that generated `#[repr(C)]` structs match
//! expected sizes and alignments on the current platform.
//!
//! These sizes correspond to the Vulkan C headers on x86_64 (pointer = 8).
//! A failure here means the generator produced a struct that disagrees with
//! the C ABI, which would corrupt every FFI call using that struct.

use std::mem::{align_of, size_of};
use vk_sys::*;

// ── ApplicationInfo ────────────────────────────────────────────────
// s_type(4) pad(4) p_next(8) p_app_name(8) app_ver(4) pad(4)
// p_engine_name(8) engine_ver(4) api_ver(4) = 48
#[test]
fn application_info_layout() {
    assert_eq!(size_of::<structs::ApplicationInfo>(), 48);
    assert_eq!(align_of::<structs::ApplicationInfo>(), 8);
}

// ── InstanceCreateInfo ─────────────────────────────────────────────
// s_type(4) pad(4) p_next(8) flags(4) pad(4) p_app_info(8)
// layer_count(4) pad(4) pp_layers(8) ext_count(4) pad(4) pp_exts(8) = 64
#[test]
fn instance_create_info_layout() {
    assert_eq!(size_of::<structs::InstanceCreateInfo>(), 64);
    assert_eq!(align_of::<structs::InstanceCreateInfo>(), 8);
}

// ── DeviceQueueCreateInfo ──────────────────────────────────────────
// s_type(4) pad(4) p_next(8) flags(4) family(4) count(4) pad(4)
// p_priorities(8) = 40
#[test]
fn device_queue_create_info_layout() {
    assert_eq!(size_of::<structs::DeviceQueueCreateInfo>(), 40);
    assert_eq!(align_of::<structs::DeviceQueueCreateInfo>(), 8);
}

// ── DeviceCreateInfo ───────────────────────────────────────────────
// s_type(4) pad(4) p_next(8) flags(4) qi_count(4) p_qis(8)
// layer_count(4) pad(4) pp_layers(8) ext_count(4) pad(4) pp_exts(8)
// p_features(8) = 72
#[test]
fn device_create_info_layout() {
    assert_eq!(size_of::<structs::DeviceCreateInfo>(), 72);
    assert_eq!(align_of::<structs::DeviceCreateInfo>(), 8);
}

// ── BufferCreateInfo ───────────────────────────────────────────────
// s_type(4) pad(4) p_next(8) flags(4) pad(4) size(8) usage(4)
// sharing_mode(4) qi_count(4) pad(4) p_qi(8) = 56
#[test]
fn buffer_create_info_layout() {
    assert_eq!(size_of::<structs::BufferCreateInfo>(), 56);
    assert_eq!(align_of::<structs::BufferCreateInfo>(), 8);
}

// ── AllocationCallbacks ────────────────────────────────────────────
// 6 pointer-sized fields: p_user_data + 5 function pointers = 48
#[test]
fn allocation_callbacks_layout() {
    assert_eq!(size_of::<structs::AllocationCallbacks>(), 48);
    assert_eq!(align_of::<structs::AllocationCallbacks>(), 8);
}

// ── Win32SurfaceCreateInfoKHR ──────────────────────────────────────
// s_type(4) pad(4) p_next(8) flags(4) pad(4) hinstance(8) hwnd(8) = 40
#[test]
fn win32_surface_create_info_layout() {
    assert_eq!(size_of::<structs::Win32SurfaceCreateInfoKHR>(), 40);
    assert_eq!(align_of::<structs::Win32SurfaceCreateInfoKHR>(), 8);
}

// ── Handle sizes ───────────────────────────────────────────────────
// Dispatchable handles are pointer-sized (usize).
// Non-dispatchable handles are always u64.
#[test]
fn dispatchable_handle_is_pointer_sized() {
    assert_eq!(size_of::<handles::Instance>(), size_of::<usize>());
    assert_eq!(size_of::<handles::PhysicalDevice>(), size_of::<usize>());
    assert_eq!(size_of::<handles::Device>(), size_of::<usize>());
    assert_eq!(size_of::<handles::Queue>(), size_of::<usize>());
    assert_eq!(size_of::<handles::CommandBuffer>(), size_of::<usize>());
}

#[test]
fn non_dispatchable_handle_is_u64() {
    assert_eq!(size_of::<handles::Buffer>(), 8);
    assert_eq!(size_of::<handles::Image>(), 8);
    assert_eq!(size_of::<handles::DeviceMemory>(), 8);
    assert_eq!(size_of::<handles::Semaphore>(), 8);
    assert_eq!(size_of::<handles::Fence>(), 8);
    assert_eq!(size_of::<handles::SurfaceKHR>(), 8);
    assert_eq!(size_of::<handles::SwapchainKHR>(), 8);
    assert_eq!(size_of::<handles::Pipeline>(), 8);
    assert_eq!(size_of::<handles::PipelineLayout>(), 8);
    assert_eq!(size_of::<handles::RenderPass>(), 8);
    assert_eq!(size_of::<handles::Framebuffer>(), 8);
    assert_eq!(size_of::<handles::ShaderModule>(), 8);
    assert_eq!(size_of::<handles::DescriptorSetLayout>(), 8);
    assert_eq!(size_of::<handles::CommandPool>(), 8);
}

// ── Enum / bitmask sizes ───────────────────────────────────────────
// Vulkan enums are i32, bitmasks are u32 (or u64 for Flags64).
#[test]
fn enum_types_are_i32() {
    assert_eq!(size_of::<enums::Result>(), 4);
    assert_eq!(size_of::<enums::StructureType>(), 4);
    assert_eq!(size_of::<enums::Format>(), 4);
    assert_eq!(size_of::<enums::ImageType>(), 4);
    assert_eq!(size_of::<enums::SharingMode>(), 4);
}

#[test]
fn bitmask_types_are_u32() {
    assert_eq!(size_of::<bitmasks::BufferUsageFlagBits>(), 4);
    assert_eq!(size_of::<bitmasks::ImageUsageFlagBits>(), 4);
    assert_eq!(size_of::<bitmasks::MemoryPropertyFlagBits>(), 4);
    assert_eq!(size_of::<bitmasks::QueueFlagBits>(), 4);
}
