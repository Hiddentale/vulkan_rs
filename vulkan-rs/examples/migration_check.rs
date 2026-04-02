#![allow(dead_code)]
// Compilation check for migrate-from-ash.md and c-to-rust.md guides.
// Exercises every vulkan_rs API call shown in those guides.
// NOT meant to be run.

use vk::bitmasks::*;
use vk::enums::*;
use vk::handles::*;
use vk::structs::*;
use vulkan_rs::Device;
use vulkan_rs::vk;

/// Verify Entry creation (migrate-from-ash Step 3)
fn check_entry_creation() {
    let loader = vulkan_rs::LibloadingLoader::new().expect("Failed to load Vulkan");
    let entry = unsafe { vulkan_rs::Entry::new(loader) }.expect("Failed to create entry");

    let app_info = ApplicationInfo::builder().api_version((1 << 22) | (3 << 12)); // Vulkan 1.3
    let create_info = InstanceCreateInfo::builder().p_application_info(&app_info);
    let _instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");
}

/// Verify Device creation (migrate-from-ash Step 3)
unsafe fn check_device_creation(instance: &vulkan_rs::Instance, physical_device: PhysicalDevice) {
    unsafe {
        let queue_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&[1.0]);
        let device_info =
            DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
        let _device = instance
            .create_device(physical_device, &device_info, None)
            .expect("Failed to create device");
    }
}

/// Verify builder pattern (no .build()), buffer creation
unsafe fn check_builder_and_buffer(device: &Device) {
    unsafe {
        let info = BufferCreateInfo::builder()
            .size(1024)
            .usage(BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(SharingMode::EXCLUSIVE);
        let buffer = device
            .create_buffer(&info, None)
            .expect("Failed to create buffer");

        let mem_req = device.get_buffer_memory_requirements(buffer);

        let alloc_info = MemoryAllocateInfo::builder()
            .allocation_size(mem_req.size)
            .memory_type_index(0);
        let memory = device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate memory");
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("Failed to bind");

        let _data = device
            .map_memory(memory, 0, 1024, MemoryMapFlags::empty())
            .expect("Failed to map");
        device.unmap_memory(memory);

        device.free_memory(memory, None);
        device.destroy_buffer(buffer, None);
    }
}

/// Verify command buffer recording (migrate-from-ash Step 5)
unsafe fn check_command_recording(device: &Device, cmd: CommandBuffer, pipeline: Pipeline) {
    unsafe {
        let begin_info =
            CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd, &begin_info)
            .expect("Failed to begin");
        device.cmd_bind_pipeline(cmd, PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.end_command_buffer(cmd).expect("Failed to end");
    }
}

/// Verify queue submit with builder deref (migrate-from-ash Step 6)
unsafe fn check_submit(
    device: &Device,
    queue: Queue,
    cmd: CommandBuffer,
    image_available: Semaphore,
    render_finished: Semaphore,
    fence: Fence,
) {
    unsafe {
        let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [cmd];
        let wait_sems = [image_available];
        let signal_sems = [render_finished];
        let submit_info = SubmitInfo::builder()
            .command_buffers(&cmd_bufs)
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_sems);
        device
            .queue_submit(queue, &[*submit_info], fence)
            .expect("Failed to submit");
    }
}

/// Verify error handling (migrate-from-ash Step 7)
unsafe fn check_error_handling(device: &Device) {
    use vk::enums::Result as VkError;

    let info = BufferCreateInfo::builder()
        .size(1024)
        .usage(BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(SharingMode::EXCLUSIVE);

    unsafe {
        match device.create_buffer(&info, None) {
            Ok(_buffer) => {}
            Err(VkError::ERROR_OUT_OF_DEVICE_MEMORY) => {}
            Err(e) => panic!("Unexpected: {e:?}"),
        }
    }
}

/// Verify extension calls (migrate-from-ash Step 8)
unsafe fn check_extensions(device: &Device) {
    unsafe {
        let info = SwapchainCreateInfoKHR::builder()
            .surface(SurfaceKHR::null())
            .min_image_count(2)
            .image_format(Format::B8G8R8A8_SRGB)
            .image_color_space(ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(Extent2D {
                width: 800,
                height: 600,
            })
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .pre_transform(SurfaceTransformFlagBitsKHR::IDENTITY)
            .composite_alpha(CompositeAlphaFlagBitsKHR::OPAQUE)
            .present_mode(PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(SwapchainKHR::null());

        // This would fail at runtime (null surface) but type-checks correctly
        let _ = device.create_swapchain_khr(&info, None);
    }
}

/// Verify pNext chain (c-to-rust.md)
unsafe fn check_pnext(instance: &vulkan_rs::Instance, physical_device: PhysicalDevice) {
    unsafe {
        let mut features12 = *PhysicalDeviceVulkan12Features::builder().buffer_device_address(true); // VkBool32
        let queue_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&[1.0]);
        let queue_infos = [*queue_info];
        let info = DeviceCreateInfo::builder()
            .push_next(&mut features12)
            .queue_create_infos(&queue_infos);
        let _ = instance.create_device(physical_device, &info, None);
    }
}

/// Verify enumerate pattern (c-to-rust.md)
unsafe fn check_enumerate(instance: &vulkan_rs::Instance) {
    unsafe {
        let _devices = instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate");
    }
}

/// Verify allocate_command_buffers returns Vec (c-to-rust.md)
unsafe fn check_allocate_cmd_buffers(device: &Device, command_pool: CommandPool) {
    unsafe {
        let alloc_info = CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);

        let cmd_buffers = device
            .allocate_command_buffers(&alloc_info)
            .expect("Failed to allocate");
        assert_eq!(cmd_buffers.len(), 2);
    }
}

/// Verify various API calls from the c-to-rust mapping table
unsafe fn check_table_calls(
    device: &Device,
    instance: &vulkan_rs::Instance,
    physical_device: PhysicalDevice,
    queue: Queue,
) {
    unsafe {
        // get_physical_device_properties
        let _props = instance.get_physical_device_properties(physical_device);

        // get_physical_device_queue_family_properties
        let _families = instance.get_physical_device_queue_family_properties(physical_device);

        // get_device_queue
        let _q = device.get_device_queue(0, 0);

        // create/destroy image
        let img_info = ImageCreateInfo::builder()
            .image_type(ImageType::_2D)
            .format(Format::R8G8B8A8_SRGB)
            .extent(Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(SampleCountFlagBits::_1)
            .tiling(ImageTiling::OPTIMAL)
            .usage(ImageUsageFlags::SAMPLED)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .initial_layout(ImageLayout::UNDEFINED);
        let _ = device.create_image(&img_info, None);

        // create/destroy fence and semaphore
        let fence_info = FenceCreateInfo::builder();
        let _ = device.create_fence(&fence_info, None);

        let sem_info = SemaphoreCreateInfo::builder();
        let _ = device.create_semaphore(&sem_info, None);

        // wait_for_fences, reset_fences
        // (can't call without real fences, but verify signatures compile)
        let fences: &[Fence] = &[];
        let _ = device.wait_for_fences(fences, true, 0);
        let _ = device.reset_fences(fences);

        // queue_present_khr
        let present_info = PresentInfoKHR::builder();
        let _ = device.queue_present_khr(queue, &present_info);

        // device_wait_idle
        let _ = device.device_wait_idle();

        // update_descriptor_sets
        let writes: &[WriteDescriptorSet] = &[];
        let copies: &[CopyDescriptorSet] = &[];
        device.update_descriptor_sets(writes, copies);
    }
}

fn main() {
    println!(
        "migration_check: all API calls from migrate-from-ash.md and c-to-rust.md compile correctly!"
    );
}
