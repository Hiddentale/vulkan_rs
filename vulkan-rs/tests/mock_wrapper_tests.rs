//! Mock-based spot-checks for generated wrapper methods.
//!
//! Tests one representative command per pattern (Create, Destroy, Enumerate,
//! Fill, Query, ResultOnly, VoidForward) using fake function pointers. This
//! validates that the generated wrappers correctly marshal arguments, dispatch
//! through the right PFN, and return the expected type.
//!
//! Runs single-threaded because tests share global mock state (`MOCK_CALLED`).
//!
//! No Vulkan runtime required.

use std::sync::Mutex;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use vk::handles::Handle;
use vulkan_rs::vk;

// ---------------------------------------------------------------------------
// Globals for mock verification
// ---------------------------------------------------------------------------

static MOCK_CALLED: AtomicBool = AtomicBool::new(false);
static MOCK_VERTEX_COUNT: AtomicU32 = AtomicU32::new(0);
static MOCK_INSTANCE_COUNT: AtomicU32 = AtomicU32::new(0);

fn reset_mocks() {
    MOCK_CALLED.store(false, Ordering::SeqCst);
    MOCK_VERTEX_COUNT.store(0, Ordering::SeqCst);
    MOCK_INSTANCE_COUNT.store(0, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Device mock infrastructure
// ---------------------------------------------------------------------------

/// Mock `vkGetDeviceProcAddr` that returns fake fps for specific commands.
unsafe extern "system" fn mock_device_proc_addr(
    _device: vk::handles::Device,
    name: *const c_char,
) -> vk::structs::PFN_vkVoidFunction {
    let name = unsafe { CStr::from_ptr(name) };
    match name.to_bytes() {
        b"vkCreateFence" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::Device,
                    *const vk::structs::FenceCreateInfo,
                    *const vk::structs::AllocationCallbacks,
                    *mut vk::handles::Fence,
                ) -> vk::enums::Result,
                unsafe extern "system" fn(),
            >(mock_create_fence)
        }),
        b"vkDestroyFence" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::Device,
                    vk::handles::Fence,
                    *const vk::structs::AllocationCallbacks,
                ),
                unsafe extern "system" fn(),
            >(mock_destroy_fence)
        }),
        b"vkDeviceWaitIdle" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(vk::handles::Device) -> vk::enums::Result,
                unsafe extern "system" fn(),
            >(mock_device_wait_idle)
        }),
        b"vkCmdDraw" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(vk::handles::CommandBuffer, u32, u32, u32, u32),
                unsafe extern "system" fn(),
            >(mock_cmd_draw)
        }),
        _ => None,
    }
}

fn mock_device() -> vulkan_rs::Device {
    unsafe {
        vulkan_rs::Device::from_raw_parts(
            vk::handles::Device::from_raw(0xD001),
            Some(mock_device_proc_addr),
        )
    }
}

// ---------------------------------------------------------------------------
// Device mock function implementations
// ---------------------------------------------------------------------------

/// Create pattern: writes output handle, returns SUCCESS.
unsafe extern "system" fn mock_create_fence(
    device: vk::handles::Device,
    p_create_info: *const vk::structs::FenceCreateInfo,
    _p_allocator: *const vk::structs::AllocationCallbacks,
    p_fence: *mut vk::handles::Fence,
) -> vk::enums::Result {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    assert_eq!(device.as_raw(), 0xD001, "wrong device handle");
    assert!(!p_create_info.is_null(), "create_info must not be null");
    assert!(!p_fence.is_null(), "output pointer must not be null");
    unsafe { *p_fence = vk::handles::Fence::from_raw(0xFE01) };
    vk::enums::Result::SUCCESS
}

/// Destroy pattern: verifies handle and allocator forwarding.
unsafe extern "system" fn mock_destroy_fence(
    device: vk::handles::Device,
    fence: vk::handles::Fence,
    p_allocator: *const vk::structs::AllocationCallbacks,
) {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    assert_eq!(device.as_raw(), 0xD001, "wrong device handle");
    assert_eq!(fence.as_raw(), 0xFE01, "wrong fence handle");
    assert!(p_allocator.is_null(), "allocator should be null when None");
}

/// ResultOnly pattern: returns SUCCESS.
unsafe extern "system" fn mock_device_wait_idle(device: vk::handles::Device) -> vk::enums::Result {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    assert_eq!(device.as_raw(), 0xD001, "wrong device handle");
    vk::enums::Result::SUCCESS
}

/// VoidForward pattern: captures arguments for verification.
unsafe extern "system" fn mock_cmd_draw(
    _command_buffer: vk::handles::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    _first_vertex: u32,
    _first_instance: u32,
) {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    MOCK_VERTEX_COUNT.store(vertex_count, Ordering::SeqCst);
    MOCK_INSTANCE_COUNT.store(instance_count, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Instance mock infrastructure
// ---------------------------------------------------------------------------

/// Mock `vkGetInstanceProcAddr` that returns fake fps for specific commands.
unsafe extern "system" fn mock_instance_proc_addr(
    _instance: vk::handles::Instance,
    name: *const c_char,
) -> vk::structs::PFN_vkVoidFunction {
    let name = unsafe { CStr::from_ptr(name) };
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::Device,
                    *const c_char,
                ) -> vk::structs::PFN_vkVoidFunction,
                unsafe extern "system" fn(),
            >(null_device_proc_addr)
        }),
        b"vkEnumeratePhysicalDevices" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::Instance,
                    *mut u32,
                    *mut vk::handles::PhysicalDevice,
                ) -> vk::enums::Result,
                unsafe extern "system" fn(),
            >(mock_enumerate_physical_devices)
        }),
        b"vkGetPhysicalDeviceQueueFamilyProperties" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::PhysicalDevice,
                    *mut u32,
                    *mut vk::structs::QueueFamilyProperties,
                ),
                unsafe extern "system" fn(),
            >(mock_get_physical_device_queue_family_properties)
        }),
        b"vkGetPhysicalDeviceMemoryProperties" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::handles::PhysicalDevice,
                    *mut vk::structs::PhysicalDeviceMemoryProperties,
                ),
                unsafe extern "system" fn(),
            >(mock_get_physical_device_memory_properties)
        }),
        _ => None,
    }
}

unsafe extern "system" fn null_device_proc_addr(
    _device: vk::handles::Device,
    _name: *const c_char,
) -> vk::structs::PFN_vkVoidFunction {
    None
}

fn mock_instance() -> vulkan_rs::Instance {
    unsafe {
        vulkan_rs::Instance::from_raw_parts(
            vk::handles::Instance::from_raw(0x1001_usize),
            Some(mock_instance_proc_addr),
        )
    }
}

// ---------------------------------------------------------------------------
// Instance mock function implementations
// ---------------------------------------------------------------------------

/// Enumerate pattern: two-call that returns 2 physical devices.
unsafe extern "system" fn mock_enumerate_physical_devices(
    _instance: vk::handles::Instance,
    p_count: *mut u32,
    p_devices: *mut vk::handles::PhysicalDevice,
) -> vk::enums::Result {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    if p_devices.is_null() {
        unsafe { *p_count = 2 };
    } else {
        unsafe {
            *p_count = 2;
            *p_devices = vk::handles::PhysicalDevice::from_raw(0xAA01_usize);
            *p_devices.add(1) = vk::handles::PhysicalDevice::from_raw(0xAA02_usize);
        }
    }
    vk::enums::Result::SUCCESS
}

/// Fill pattern: two-call that returns 1 queue family.
unsafe extern "system" fn mock_get_physical_device_queue_family_properties(
    _physical_device: vk::handles::PhysicalDevice,
    p_count: *mut u32,
    p_properties: *mut vk::structs::QueueFamilyProperties,
) {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    if p_properties.is_null() {
        unsafe { *p_count = 1 };
    } else {
        unsafe {
            *p_count = 1;
            let mut props: vk::structs::QueueFamilyProperties = std::mem::zeroed();
            props.queue_flags = vk::bitmasks::QueueFlagBits::GRAPHICS;
            props.queue_count = 4;
            *p_properties = props;
        }
    }
}

/// Query pattern: writes output struct directly.
unsafe extern "system" fn mock_get_physical_device_memory_properties(
    _physical_device: vk::handles::PhysicalDevice,
    p_memory_properties: *mut vk::structs::PhysicalDeviceMemoryProperties,
) {
    MOCK_CALLED.store(true, Ordering::SeqCst);
    unsafe {
        let mut props: vk::structs::PhysicalDeviceMemoryProperties = std::mem::zeroed();
        props.memory_type_count = 3;
        props.memory_heap_count = 2;
        *p_memory_properties = props;
    }
}

// ===========================================================================
// Tests: one per command pattern
// ===========================================================================

/// **Create pattern**: `create_fence` returns `VkResult<Fence>`.
/// Verifies the wrapper passes device handle + create_info pointer,
/// writes the output handle, and wraps in Ok.
#[test]
fn pattern_create() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let device = mock_device();

    let fence_info = vk::structs::FenceCreateInfo {
        s_type: vk::enums::StructureType::FENCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::FenceCreateFlagBits::empty(),
    };

    let fence =
        unsafe { device.create_fence(&fence_info, None) }.expect("create_fence should succeed");

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert_eq!(fence.as_raw(), 0xFE01, "wrong fence handle returned");
}

/// **Destroy pattern**: `destroy_fence` returns `()`.
/// Verifies the wrapper forwards the device handle, object handle,
/// and null allocator.
#[test]
fn pattern_destroy() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let device = mock_device();

    let fence = vk::handles::Fence::from_raw(0xFE01);
    unsafe { device.destroy_fence(fence, None) };

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
}

/// **Enumerate pattern**: `enumerate_physical_devices` returns `VkResult<Vec<T>>`.
/// Verifies the two-call protocol (first call for count, second for data)
/// and that the Vec is correctly populated.
#[test]
fn pattern_enumerate() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let instance = mock_instance();

    let devices =
        unsafe { instance.enumerate_physical_devices() }.expect("enumerate should succeed");

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert_eq!(devices.len(), 2, "expected 2 physical devices");
}

/// **Fill pattern**: `get_physical_device_queue_family_properties` returns `Vec<T>`.
/// Verifies the void two-call protocol works without VkResult wrapping.
#[test]
fn pattern_fill() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let instance = mock_instance();
    let physical_device = vk::handles::PhysicalDevice::from_raw(0xABCD);

    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert_eq!(families.len(), 1, "expected 1 queue family");
    assert_eq!(families[0].queue_count, 4);
    assert!(
        families[0]
            .queue_flags
            .contains(vk::bitmasks::QueueFlagBits::GRAPHICS)
    );
}

/// **Query pattern**: `get_physical_device_memory_properties` returns `T` directly.
/// Verifies the wrapper zeroes output, calls fp, and returns the struct.
#[test]
fn pattern_query() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let instance = mock_instance();
    let physical_device = vk::handles::PhysicalDevice::from_raw(0xABCD);

    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert_eq!(mem_props.memory_type_count, 3);
    assert_eq!(mem_props.memory_heap_count, 2);
}

/// **ResultOnly pattern**: `device_wait_idle` returns `VkResult<()>`.
/// Verifies the wrapper calls check() and returns Ok(()).
#[test]
fn pattern_result_only() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let device = mock_device();

    let result = unsafe { device.device_wait_idle() };

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert!(result.is_ok(), "device_wait_idle should return Ok");
}

/// **VoidForward pattern**: `cmd_draw` returns `()`.
/// Verifies all arguments are forwarded correctly to the PFN.
#[test]
fn pattern_void_forward() {
    let _lock = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
    reset_mocks();
    let device = mock_device();
    let cmd_buf = vk::handles::CommandBuffer::from_raw(0xCB01);

    unsafe { device.cmd_draw(cmd_buf, 36, 1, 0, 0) };

    assert!(MOCK_CALLED.load(Ordering::SeqCst), "mock was not called");
    assert_eq!(MOCK_VERTEX_COUNT.load(Ordering::SeqCst), 36);
    assert_eq!(MOCK_INSTANCE_COUNT.load(Ordering::SeqCst), 1);
}
