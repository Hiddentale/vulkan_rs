//! Runtime integration tests for generated wrapper methods.
//!
//! These exercise the auto-generated `impl Device { ... }` and
//! `impl Instance { ... }` methods against a real Vulkan driver.
//! All tests require a Vulkan runtime and are `#[ignore]`d by default.
//!
//! Run with: `cargo test -p vk-engine --test generated_wrappers_integration -- --ignored`

use vk::handles::Handle;
use vk_engine::{Entry, Instance, LibloadingLoader, vk};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn create_entry() -> Entry {
    let loader = LibloadingLoader::new().expect("failed to load Vulkan library");
    unsafe { Entry::new(loader) }.expect("failed to create Entry")
}

fn create_instance() -> (Entry, Instance) {
    let entry = create_entry();
    let app_info = vk::structs::ApplicationInfo {
        s_type: vk::enums::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: std::ptr::null(),
        application_version: 0,
        p_engine_name: std::ptr::null(),
        engine_version: 0,
        api_version: 1u32 << 22,
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
    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("failed to create instance");
    (entry, instance)
}

struct TestDevice {
    _entry: Entry,
    instance: Instance,
    device: vk_engine::Device,
    physical_device: vk::handles::PhysicalDevice,
    queue_family: u32,
}

impl TestDevice {
    fn new() -> Self {
        let (entry, instance) = create_instance();

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("failed to enumerate physical devices");
        let physical_device = physical_devices[0];

        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family = families
            .iter()
            .position(|f| {
                f.queue_flags
                    .contains(vk::bitmasks::QueueFlagBits::GRAPHICS)
            })
            .expect("no graphics queue family") as u32;

        let queue_priority = 1.0f32;
        let queue_create_info = vk::structs::DeviceQueueCreateInfo {
            s_type: vk::enums::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::bitmasks::DeviceQueueCreateFlagBits::empty(),
            queue_family_index: queue_family,
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

        Self {
            _entry: entry,
            instance,
            device,
            physical_device,
            queue_family,
        }
    }
}

impl Drop for TestDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Instance-level generated wrapper tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn enumerate_device_extension_properties() {
    let (_entry, instance) = create_instance();
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap();

    let extensions =
        unsafe { instance.enumerate_device_extension_properties(devices[0], std::ptr::null()) }
            .expect("enumerate_device_extension_properties failed");

    println!("Device extensions: {}", extensions.len());
    assert!(
        !extensions.is_empty(),
        "expected at least one device extension"
    );

    unsafe { instance.destroy_instance(None) };
}

#[test]
#[ignore]
fn get_physical_device_memory_properties() {
    let (_entry, instance) = create_instance();
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap();

    let mem_props = unsafe { instance.get_physical_device_memory_properties(devices[0]) };

    assert!(
        mem_props.memory_type_count > 0,
        "expected at least one memory type"
    );
    assert!(
        mem_props.memory_heap_count > 0,
        "expected at least one memory heap"
    );
    println!(
        "Memory: {} types, {} heaps",
        mem_props.memory_type_count, mem_props.memory_heap_count
    );

    unsafe { instance.destroy_instance(None) };
}

#[test]
#[ignore]
fn get_physical_device_features() {
    let (_entry, instance) = create_instance();
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap();

    // Just verify the call completes without crashing.
    let _features = unsafe { instance.get_physical_device_features(devices[0]) };

    unsafe { instance.destroy_instance(None) };
}

// ---------------------------------------------------------------------------
// Device-level generated wrapper tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn create_and_destroy_buffer() {
    let t = TestDevice::new();

    let buffer_info = vk::structs::BufferCreateInfo {
        s_type: vk::enums::StructureType::BUFFER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::BufferCreateFlagBits::empty(),
        size: 1024,
        usage: vk::bitmasks::BufferUsageFlagBits::VERTEX_BUFFER,
        sharing_mode: vk::enums::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
    };

    let buffer =
        unsafe { t.device.create_buffer(&buffer_info, None) }.expect("create_buffer failed");
    assert!(!buffer.is_null());

    unsafe { t.device.destroy_buffer(buffer, None) };
}

#[test]
#[ignore]
fn get_buffer_memory_requirements() {
    let t = TestDevice::new();

    let buffer_info = vk::structs::BufferCreateInfo {
        s_type: vk::enums::StructureType::BUFFER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::BufferCreateFlagBits::empty(),
        size: 256,
        usage: vk::bitmasks::BufferUsageFlagBits::UNIFORM_BUFFER,
        sharing_mode: vk::enums::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
    };

    let buffer = unsafe { t.device.create_buffer(&buffer_info, None) }.unwrap();
    let mem_reqs = unsafe { t.device.get_buffer_memory_requirements(buffer) };

    assert!(mem_reqs.size >= 256, "memory requirements size too small");
    assert!(mem_reqs.alignment > 0, "alignment must be non-zero");
    println!(
        "Buffer memory: size={}, alignment={}, memory_type_bits=0x{:x}",
        mem_reqs.size, mem_reqs.alignment, mem_reqs.memory_type_bits
    );

    unsafe { t.device.destroy_buffer(buffer, None) };
}

#[test]
#[ignore]
fn allocate_and_free_memory() {
    let t = TestDevice::new();

    let mem_props = unsafe {
        t.instance
            .get_physical_device_memory_properties(t.physical_device)
    };

    // Find a host-visible memory type.
    let memory_type_index = (0..mem_props.memory_type_count)
        .find(|&i| {
            mem_props.memory_types[i as usize]
                .property_flags
                .contains(vk::bitmasks::MemoryPropertyFlagBits::HOST_VISIBLE)
        })
        .expect("no host-visible memory type");

    let alloc_info = vk::structs::MemoryAllocateInfo {
        s_type: vk::enums::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        allocation_size: 4096,
        memory_type_index,
    };

    let memory =
        unsafe { t.device.allocate_memory(&alloc_info, None) }.expect("allocate_memory failed");
    assert!(!memory.is_null());

    unsafe { t.device.free_memory(memory, None) };
}

#[test]
#[ignore]
fn create_and_destroy_fence() {
    let t = TestDevice::new();

    let fence_info = vk::structs::FenceCreateInfo {
        s_type: vk::enums::StructureType::FENCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::FenceCreateFlagBits::SIGNALED,
    };

    let fence = unsafe { t.device.create_fence(&fence_info, None) }.expect("create_fence failed");
    assert!(!fence.is_null());

    // Wait for signaled fence should succeed immediately.
    let result = unsafe { t.device.wait_for_fences(&[fence], 1, 0) };
    assert!(result.is_ok(), "wait_for_fences failed: {result:?}");

    // Reset and verify it's unsignaled.
    unsafe { t.device.reset_fences(&[fence]) }.expect("reset_fences failed");

    unsafe { t.device.destroy_fence(fence, None) };
}

#[test]
#[ignore]
fn create_and_destroy_semaphore() {
    let t = TestDevice::new();

    let sem_info = vk::structs::SemaphoreCreateInfo {
        s_type: vk::enums::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::SemaphoreCreateFlagBits::empty(),
    };

    let semaphore =
        unsafe { t.device.create_semaphore(&sem_info, None) }.expect("create_semaphore failed");
    assert!(!semaphore.is_null());

    unsafe { t.device.destroy_semaphore(semaphore, None) };
}

#[test]
#[ignore]
fn create_command_pool_and_submit_empty_buffer() {
    let t = TestDevice::new();

    // Create command pool.
    let pool_info = vk::structs::CommandPoolCreateInfo {
        s_type: vk::enums::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::CommandPoolCreateFlagBits::RESET_COMMAND_BUFFER,
        queue_family_index: t.queue_family,
    };
    let pool = unsafe { t.device.create_command_pool(&pool_info, None) }
        .expect("create_command_pool failed");

    // Allocate command buffer (uses raw forward — count is inside the struct).
    let alloc_info = vk::structs::CommandBufferAllocateInfo {
        s_type: vk::enums::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: std::ptr::null(),
        command_pool: pool,
        level: vk::enums::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };
    let mut cmd_buf = vk::handles::CommandBuffer::null();
    unsafe {
        let fp = t
            .device
            .commands()
            .allocate_command_buffers
            .expect("not loaded");
        let result = fp(t.device.handle(), &alloc_info, &mut cmd_buf);
        assert!(
            result.as_raw() >= 0,
            "allocate_command_buffers failed: {:?}",
            result
        );
    };
    assert!(!cmd_buf.is_null());

    // Begin → End (empty command buffer).
    let begin_info = vk::structs::CommandBufferBeginInfo {
        s_type: vk::enums::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::CommandBufferUsageFlagBits::ONE_TIME_SUBMIT,
        p_inheritance_info: std::ptr::null(),
    };
    unsafe { t.device.begin_command_buffer(cmd_buf, &begin_info) }
        .expect("begin_command_buffer failed");
    unsafe { t.device.end_command_buffer(cmd_buf) }.expect("end_command_buffer failed");

    // Submit.
    let queue = unsafe { t.device.get_device_queue(t.queue_family, 0) };
    let submit_info = vk::structs::SubmitInfo {
        s_type: vk::enums::StructureType::SUBMIT_INFO,
        p_next: std::ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: std::ptr::null(),
        p_wait_dst_stage_mask: std::ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd_buf,
        signal_semaphore_count: 0,
        p_signal_semaphores: std::ptr::null(),
    };

    let fence_info = vk::structs::FenceCreateInfo {
        s_type: vk::enums::StructureType::FENCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::FenceCreateFlagBits::empty(),
    };
    let fence = unsafe { t.device.create_fence(&fence_info, None) }.unwrap();

    unsafe { t.device.queue_submit(queue, &[submit_info], fence) }.expect("queue_submit failed");

    // Wait for completion.
    unsafe { t.device.wait_for_fences(&[fence], 1, u64::MAX) }.expect("wait_for_fences failed");

    // Cleanup.
    unsafe {
        t.device.destroy_fence(fence, None);
        t.device.free_command_buffers(pool, &[cmd_buf]);
        t.device.destroy_command_pool(pool, None);
    };
}

#[test]
#[ignore]
fn create_and_destroy_image() {
    let t = TestDevice::new();

    let image_info = vk::structs::ImageCreateInfo {
        s_type: vk::enums::StructureType::IMAGE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::ImageCreateFlagBits::empty(),
        image_type: vk::enums::ImageType::_2D,
        format: vk::enums::Format::R8G8B8A8_UNORM,
        extent: vk::structs::Extent3D {
            width: 64,
            height: 64,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::bitmasks::SampleCountFlagBits::_1,
        tiling: vk::enums::ImageTiling::OPTIMAL,
        usage: vk::bitmasks::ImageUsageFlagBits::SAMPLED,
        sharing_mode: vk::enums::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        initial_layout: vk::enums::ImageLayout::UNDEFINED,
    };

    let image = unsafe { t.device.create_image(&image_info, None) }.expect("create_image failed");
    assert!(!image.is_null());

    let mem_reqs = unsafe { t.device.get_image_memory_requirements(image) };
    assert!(
        mem_reqs.size > 0,
        "image memory requirements size must be > 0"
    );

    unsafe { t.device.destroy_image(image, None) };
}

#[test]
#[ignore]
fn create_and_destroy_sampler() {
    let t = TestDevice::new();

    let sampler_info = vk::structs::SamplerCreateInfo {
        s_type: vk::enums::StructureType::SAMPLER_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::SamplerCreateFlagBits::empty(),
        mag_filter: vk::enums::Filter::LINEAR,
        min_filter: vk::enums::Filter::LINEAR,
        mipmap_mode: vk::enums::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::enums::SamplerAddressMode::REPEAT,
        address_mode_v: vk::enums::SamplerAddressMode::REPEAT,
        address_mode_w: vk::enums::SamplerAddressMode::REPEAT,
        mip_lod_bias: 0.0,
        anisotropy_enable: 0,
        max_anisotropy: 1.0,
        compare_enable: 0,
        compare_op: vk::enums::CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: vk::enums::BorderColor::INT_OPAQUE_BLACK,
        unnormalized_coordinates: 0,
    };

    let sampler =
        unsafe { t.device.create_sampler(&sampler_info, None) }.expect("create_sampler failed");
    assert!(!sampler.is_null());

    unsafe { t.device.destroy_sampler(sampler, None) };
}

#[test]
#[ignore]
fn create_and_destroy_pipeline_layout() {
    let t = TestDevice::new();

    let layout_info = vk::structs::PipelineLayoutCreateInfo {
        s_type: vk::enums::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::bitmasks::PipelineLayoutCreateFlagBits::empty(),
        set_layout_count: 0,
        p_set_layouts: std::ptr::null(),
        push_constant_range_count: 0,
        p_push_constant_ranges: std::ptr::null(),
    };

    let layout = unsafe { t.device.create_pipeline_layout(&layout_info, None) }
        .expect("create_pipeline_layout failed");
    assert!(!layout.is_null());

    unsafe { t.device.destroy_pipeline_layout(layout, None) };
}
