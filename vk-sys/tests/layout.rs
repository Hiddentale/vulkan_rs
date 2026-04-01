//! Struct layout tests,verify that generated `#[repr(C)]` structs match
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

#[test]
fn bitmask_64bit_types_are_u64() {
    assert_eq!(size_of::<bitmasks::AccessFlagBits2>(), 8);
    assert_eq!(size_of::<bitmasks::PipelineStageFlagBits2>(), 8);
    assert_eq!(size_of::<bitmasks::FormatFeatureFlagBits2>(), 8);
    assert_eq!(size_of::<bitmasks::PipelineCreateFlagBits2>(), 8);
    assert_eq!(size_of::<bitmasks::BufferUsageFlagBits2>(), 8);
    assert_eq!(size_of::<bitmasks::MemoryDecompressionMethodFlagBitsEXT>(), 8);
}

// ── Struct with 64-bit bitmask fields ─────────────────────────────
// MemoryBarrier2: s_type(4) pad(4) p_next(8) src_stage(8) src_access(8)
//                 dst_stage(8) dst_access(8) = 48
#[test]
fn memory_barrier2_layout() {
    assert_eq!(size_of::<structs::MemoryBarrier2>(), 48);
    assert_eq!(align_of::<structs::MemoryBarrier2>(), 8);
}

// ── 64-bit bitmask aliases match their targets ────────────────────
// KHR/NV aliases of 64-bit flags must be the same size as the core type.
#[test]
fn bitmask_64bit_aliases_match_target_size() {
    assert_eq!(
        size_of::<structs::PipelineCreateFlags2KHR>(),
        size_of::<bitmasks::PipelineCreateFlagBits2>(),
    );
    assert_eq!(
        size_of::<structs::PipelineStageFlags2KHR>(),
        size_of::<bitmasks::PipelineStageFlagBits2>(),
    );
    assert_eq!(
        size_of::<structs::MemoryDecompressionMethodFlagsNV>(),
        size_of::<bitmasks::MemoryDecompressionMethodFlagBitsEXT>(),
    );
}

// ── High-bit constants for 64-bit bitmasks ────────────────────────
// At least one constant must exceed u32::MAX, proving no silent truncation.
#[test]
fn bitmask_64bit_has_high_bit_constants() {
    // AccessFlagBits2: SHADER_SAMPLED_READ is bit 32
    assert!(
        bitmasks::AccessFlagBits2::_2_SHADER_SAMPLED_READ.as_raw() > u32::MAX as u64,
        "AccessFlagBits2 should have constants above u32::MAX"
    );
    // PipelineStageFlagBits2: COPY is bit 32
    assert!(
        bitmasks::PipelineStageFlagBits2::_2_COPY.as_raw() > u32::MAX as u64,
        "PipelineStageFlagBits2 should have constants above u32::MAX"
    );
}

// ── Union layouts ─────────────────────────────────────────────────
// Union size = largest member, alignment = most-aligned member.

// ClearColorValue: float32[4] | int32[4] | uint32[4],all 16 bytes, align 4
#[test]
fn clear_color_value_layout() {
    assert_eq!(size_of::<structs::ClearColorValue>(), 16);
    assert_eq!(align_of::<structs::ClearColorValue>(), 4);
}

// ClearValue: ClearColorValue(16) | ClearDepthStencilValue(8) = 16, align 4
#[test]
fn clear_value_layout() {
    assert_eq!(size_of::<structs::ClearValue>(), 16);
    assert_eq!(align_of::<structs::ClearValue>(), 4);
}

// DeviceOrHostAddressKHR: u64 | *mut c_void = 8, align 8
#[test]
fn device_or_host_address_layout() {
    assert_eq!(size_of::<structs::DeviceOrHostAddressKHR>(), 8);
    assert_eq!(align_of::<structs::DeviceOrHostAddressKHR>(), 8);
}

// PerformanceCounterResultKHR: i32|i64|u32|u64|f32|f64 = 8, align 8
#[test]
fn performance_counter_result_layout() {
    assert_eq!(size_of::<structs::PerformanceCounterResultKHR>(), 8);
    assert_eq!(align_of::<structs::PerformanceCounterResultKHR>(), 8);
}
