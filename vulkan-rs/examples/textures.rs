// How-To: Load and Sample Textures
// Based on hello_triangle_4.rs, modified to load a texture from disk,
// upload it to the GPU, and sample it in a fragment shader.
// <https://hiddentale.github.io/vulkan_rs/how-to/textures.html>

use vk::bitmasks::*;
use vk::enums::*;
use vk::handles::*;
use vk::structs::*;
use vulkan_rs::vk;
use vulkan_rs::{Device, Entry, LibloadingLoader, cast_to_u32};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    state: Option<VulkanState>,
}

struct VulkanState {
    window: Window,
    instance: vulkan_rs::Instance,
    device: Device,
    graphics_queue: Queue,
    swapchain: SwapchainKHR,
    extent: Extent2D,
    render_pass: RenderPass,
    pipeline: Pipeline,
    pipeline_layout: PipelineLayout,
    framebuffers: Vec<Framebuffer>,
    image_views: Vec<ImageView>,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    image_available: Semaphore,
    render_finished: Semaphore,
    in_flight_fence: Fence,
    surface: SurfaceKHR,
    // Texture resources
    texture_image: Image,
    texture_memory: DeviceMemory,
    texture_view: ImageView,
    sampler: Sampler,
    descriptor_set_layout: DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: DescriptorSet,
    // Staging buffer (cleaned up after upload, but keep memory handle)
    staging_buffer: Buffer,
    staging_memory: DeviceMemory,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Textures")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        self.state = Some(init_vulkan(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(state) = &self.state {
                    unsafe { state.device.device_wait_idle() }.expect("Failed to wait idle");
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &self.state {
                    unsafe { draw_frame(state) };
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App { state: None };
    event_loop.run_app(&mut app).expect("Event loop error");

    if let Some(state) = app.state {
        unsafe { cleanup(state) };
    }
}

/// Find a memory type index that satisfies requirements and desired properties.
fn find_memory_type(
    mem_properties: &PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
        {
            return i;
        }
    }
    panic!("Failed to find suitable memory type");
}

#[allow(clippy::too_many_arguments)]
fn init_vulkan(window: Window) -> VulkanState {
    // ════════════════════════════════════════════════════════════
    // Standard setup (same as hello_triangle_4)
    // ════════════════════════════════════════════════════════════
    let loader = LibloadingLoader::new().expect("Vulkan library not found");
    let entry = unsafe { Entry::new(loader) }.expect("Failed to load Vulkan");

    let version = entry.version().expect("Failed to query version");
    println!(
        "Vulkan {}.{}.{}",
        version.major, version.minor, version.patch
    );

    let surface_extensions = vulkan_rs::required_extensions();
    let extension_ptrs: Vec<*const i8> =
        surface_extensions.iter().map(|ext| ext.as_ptr()).collect();

    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    let layer_ptrs = [validation_layer.as_ptr()];

    let app_info = ApplicationInfo::builder()
        .p_application_name(c"Textures".as_ptr())
        .application_version(1)
        .p_engine_name(c"No Engine".as_ptr())
        .engine_version(1)
        .api_version(1 << 22);

    let create_info = InstanceCreateInfo::builder()
        .p_application_info(&*app_info)
        .enabled_extension_count(extension_ptrs.len() as u32)
        .pp_enabled_extension_names(extension_ptrs.as_ptr())
        .enabled_layer_count(layer_ptrs.len() as u32)
        .pp_enabled_layer_names(layer_ptrs.as_ptr());

    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");

    let surface = unsafe { instance.create_surface(&window, &window, None) }
        .expect("Failed to create surface");

    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.expect("Failed to enumerate GPUs");

    let mut physical_device = PhysicalDevice::null();
    let mut graphics_family_index = 0u32;

    'outer: for &pd in &physical_devices {
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        for (i, family) in families.iter().enumerate() {
            let graphics = family.queue_flags & QueueFlags::GRAPHICS != QueueFlags::empty();
            let present =
                unsafe { instance.get_physical_device_surface_support_khr(pd, i as u32, surface) }
                    .unwrap_or(0)
                    != 0;
            if graphics && present {
                physical_device = pd;
                graphics_family_index = i as u32;
                break 'outer;
            }
        }
    }
    assert!(!physical_device.is_null(), "No suitable GPU found");

    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let device_extensions = [vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr()];
    let queue_priority = 1.0_f32;
    let queue_info = DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));
    let device_info = DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&*queue_info))
        .enabled_extension_count(device_extensions.len() as u32)
        .pp_enabled_extension_names(device_extensions.as_ptr());

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .expect("Failed to create device");
    let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

    let caps =
        unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }
            .expect("Failed to query surface capabilities");
    let formats =
        unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface) }
            .expect("Failed to query surface formats");
    let present_modes =
        unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface) }
            .expect("Failed to query present modes");

    let surface_format = formats
        .iter()
        .find(|f| {
            f.format == Format::B8G8R8A8_SRGB && f.color_space == ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(&formats[0]);
    let present_mode = if present_modes.contains(&PresentModeKHR::MAILBOX) {
        PresentModeKHR::MAILBOX
    } else {
        PresentModeKHR::FIFO
    };
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        let size = window.inner_size();
        Extent2D {
            width: size
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: size
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    let image_count = {
        let desired = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            desired.min(caps.max_image_count)
        } else {
            desired
        }
    };

    let swapchain_info = SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(CompositeAlphaFlagBitsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(SwapchainKHR::null());

    let swapchain = unsafe { device.create_swapchain_khr(&swapchain_info, None) }
        .expect("Failed to create swapchain");

    let images = unsafe { device.get_swapchain_images_khr(swapchain) }
        .expect("Failed to get swapchain images");

    let image_views: Vec<ImageView> = images
        .iter()
        .map(|&image| {
            let info = ImageViewCreateInfo::builder()
                .image(image)
                .view_type(ImageViewType::_2D)
                .format(surface_format.format)
                .components(ComponentMapping {
                    r: ComponentSwizzle::IDENTITY,
                    g: ComponentSwizzle::IDENTITY,
                    b: ComponentSwizzle::IDENTITY,
                    a: ComponentSwizzle::IDENTITY,
                })
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe { device.create_image_view(&info, None) }.expect("Failed to create image view")
        })
        .collect();

    // ════════════════════════════════════════════════════════════
    // Texture upload (textures.md Steps 1-7)
    // ════════════════════════════════════════════════════════════

    // Step 1: Load pixels from disk
    let img = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/assets/test_texture.png"
    ))
    .expect("Failed to open image")
    .to_rgba8();

    let (width, height) = img.dimensions();
    let pixels = img.as_raw();
    let image_size = (width * height * 4) as u64; // 4 bytes per RGBA pixel

    // Step 2: Create a staging buffer
    let staging_info = BufferCreateInfo::builder()
        .size(image_size)
        .usage(BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe { device.create_buffer(&staging_info, None) }
        .expect("Failed to create staging buffer");
    let staging_reqs = unsafe { device.get_buffer_memory_requirements(staging_buffer) };

    let staging_alloc = MemoryAllocateInfo::builder()
        .allocation_size(staging_reqs.size)
        .memory_type_index(find_memory_type(
            &mem_properties,
            staging_reqs.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        ));
    let staging_memory = unsafe { device.allocate_memory(&staging_alloc, None) }
        .expect("Failed to allocate staging memory");
    unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
        .expect("Failed to bind staging buffer");

    // Map, copy pixels, unmap.
    unsafe {
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        device
            .map_memory(
                staging_memory,
                0,
                image_size,
                MemoryMapFlags::empty(),
                &mut ptr,
            )
            .expect("Failed to map memory");
        core::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, image_size as usize);
        device.unmap_memory(staging_memory);
    }

    // Step 3: Create the device-local image
    let image_info = ImageCreateInfo::builder()
        .image_type(ImageType::_2D)
        .format(Format::R8G8B8A8_SRGB)
        .extent(Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(SampleCountFlagBits::_1)
        .tiling(ImageTiling::OPTIMAL)
        .usage(ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED)
        .sharing_mode(SharingMode::EXCLUSIVE)
        .initial_layout(ImageLayout::UNDEFINED);

    let texture_image =
        unsafe { device.create_image(&image_info, None) }.expect("Failed to create image");

    let img_reqs = unsafe { device.get_image_memory_requirements(texture_image) };
    let tex_alloc = MemoryAllocateInfo::builder()
        .allocation_size(img_reqs.size)
        .memory_type_index(find_memory_type(
            &mem_properties,
            img_reqs.memory_type_bits,
            MemoryPropertyFlags::DEVICE_LOCAL,
        ));
    let texture_memory = unsafe { device.allocate_memory(&tex_alloc, None) }
        .expect("Failed to allocate texture memory");
    unsafe { device.bind_image_memory(texture_image, texture_memory, 0) }
        .expect("Failed to bind image memory");

    // Steps 4-6: One-shot command buffer for layout transitions and copy
    let pool_info = CommandPoolCreateInfo::builder()
        .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(graphics_family_index);
    let command_pool =
        unsafe { device.create_command_pool(&pool_info, None) }.expect("Failed to create pool");

    let alloc_info = CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let upload_cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
        .expect("Failed to allocate upload command buffer")[0];

    unsafe {
        let begin_info =
            CommandBufferBeginInfo::builder().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(upload_cmd, &begin_info)
            .expect("Failed to begin upload cmd");

        // Step 4: Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
        let barrier_to_transfer = ImageMemoryBarrier::builder()
            .old_layout(ImageLayout::UNDEFINED)
            .new_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::constants::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::constants::QUEUE_FAMILY_IGNORED)
            .image(texture_image)
            .subresource_range(ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(AccessFlags::NONE)
            .dst_access_mask(AccessFlags::TRANSFER_WRITE);

        device.cmd_pipeline_barrier(
            upload_cmd,
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            DependencyFlags::empty(),
            &[],
            &[],
            &[*barrier_to_transfer],
        );

        // Step 5: Copy staging buffer to image
        let region = BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: ImageSubresourceLayers {
                aspect_mask: ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: Offset3D { x: 0, y: 0, z: 0 },
            image_extent: Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        device.cmd_copy_buffer_to_image(
            upload_cmd,
            staging_buffer,
            texture_image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Step 6: Transition TRANSFER_DST -> SHADER_READ_ONLY
        let barrier_to_shader = ImageMemoryBarrier::builder()
            .old_layout(ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::constants::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::constants::QUEUE_FAMILY_IGNORED)
            .image(texture_image)
            .subresource_range(ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ);

        device.cmd_pipeline_barrier(
            upload_cmd,
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::FRAGMENT_SHADER,
            DependencyFlags::empty(),
            &[],
            &[],
            &[*barrier_to_shader],
        );

        device
            .end_command_buffer(upload_cmd)
            .expect("Failed to end upload cmd");

        // Submit and wait
        let cmd_bufs = [upload_cmd];
        let submit = SubmitInfo::builder().command_buffers(&cmd_bufs);
        device
            .queue_submit(graphics_queue, &[*submit], Fence::null())
            .expect("Failed to submit upload");
        device
            .queue_wait_idle(graphics_queue)
            .expect("Failed to wait for upload");
    }

    // Step 7: Create image view and sampler
    let view_info = ImageViewCreateInfo::builder()
        .image(texture_image)
        .view_type(ImageViewType::_2D)
        .format(Format::R8G8B8A8_SRGB)
        .subresource_range(ImageSubresourceRange {
            aspect_mask: ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let texture_view = unsafe { device.create_image_view(&view_info, None) }
        .expect("Failed to create texture image view");

    let sampler_info = SamplerCreateInfo::builder()
        .mag_filter(Filter::LINEAR)
        .min_filter(Filter::LINEAR)
        .address_mode_u(SamplerAddressMode::REPEAT)
        .address_mode_v(SamplerAddressMode::REPEAT)
        .address_mode_w(SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .border_color(BorderColor::INT_OPAQUE_BLACK)
        .mipmap_mode(SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(0.0);

    let sampler =
        unsafe { device.create_sampler(&sampler_info, None) }.expect("Failed to create sampler");

    // ════════════════════════════════════════════════════════════
    // Descriptor set layout, pool, and set (Step 8)
    // ════════════════════════════════════════════════════════════
    let binding = DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::FRAGMENT,
        p_immutable_samplers: core::ptr::null(),
    };
    let layout_bindings = [binding];
    let dsl_info = DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&dsl_info, None) }
        .expect("Failed to create descriptor set layout");

    let pool_size = DescriptorPoolSize {
        r#type: DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    };
    let pool_sizes = [pool_size];
    let dp_info = DescriptorPoolCreateInfo::builder()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
        .expect("Failed to create descriptor pool");

    let set_layouts = [descriptor_set_layout];
    let ds_alloc = DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&ds_alloc) }
        .expect("Failed to allocate descriptor set")[0];

    // Step 8: Update descriptor set
    let image_descriptor = DescriptorImageInfo {
        sampler,
        image_view: texture_view,
        image_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };

    let image_infos = [image_descriptor];
    let write = WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos);

    unsafe { device.update_descriptor_sets(&[*write], &[]) };

    // ════════════════════════════════════════════════════════════
    // Render pass, pipeline (using textured shaders)
    // ════════════════════════════════════════════════════════════
    let vert_bytes = include_bytes!("shaders/textured.vert.spv");
    let frag_bytes = include_bytes!("shaders/textured.frag.spv");
    let vert_code = cast_to_u32(vert_bytes).expect("Vertex SPIR-V not aligned");
    let frag_code = cast_to_u32(frag_bytes).expect("Fragment SPIR-V not aligned");

    let vert_info = ShaderModuleCreateInfo::builder()
        .code_size(vert_code.len() * 4)
        .p_code(vert_code.as_ptr());
    let frag_info = ShaderModuleCreateInfo::builder()
        .code_size(frag_code.len() * 4)
        .p_code(frag_code.as_ptr());

    let vert_module = unsafe { device.create_shader_module(&vert_info, None) }
        .expect("Failed to create vertex shader module");
    let frag_module = unsafe { device.create_shader_module(&frag_info, None) }
        .expect("Failed to create fragment shader module");

    let color_attachment = AttachmentDescription {
        flags: AttachmentDescriptionFlags::empty(),
        format: surface_format.format,
        samples: SampleCountFlagBits::_1,
        load_op: AttachmentLoadOp::CLEAR,
        store_op: AttachmentStoreOp::STORE,
        stencil_load_op: AttachmentLoadOp::DONT_CARE,
        stencil_store_op: AttachmentStoreOp::DONT_CARE,
        initial_layout: ImageLayout::UNDEFINED,
        final_layout: ImageLayout::PRESENT_SRC,
    };
    let color_ref = AttachmentReference {
        attachment: 0,
        layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = SubpassDescription {
        flags: SubpassDescriptionFlags::empty(),
        pipeline_bind_point: PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: core::ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: &color_ref,
        p_resolve_attachments: core::ptr::null(),
        p_depth_stencil_attachment: core::ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: core::ptr::null(),
    };
    let dependency = SubpassDependency {
        src_subpass: vk::constants::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: AccessFlags::NONE,
        dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: DependencyFlags::empty(),
    };

    let render_pass_info = RenderPassCreateInfo::builder()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    let render_pass = unsafe { device.create_render_pass(&render_pass_info, None) }
        .expect("Failed to create render pass");

    let set_layouts_for_pipeline = [descriptor_set_layout];
    let layout_info = PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts_for_pipeline);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .expect("Failed to create pipeline layout");

    let entry_name = c"main";
    let stages = [
        *PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vert_module)
            .p_name(entry_name.as_ptr()),
        *PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .p_name(entry_name.as_ptr()),
    ];
    let vertex_input = PipelineVertexInputStateCreateInfo::builder();
    let input_assembly =
        PipelineInputAssemblyStateCreateInfo::builder().topology(PrimitiveTopology::TRIANGLE_LIST);
    let mut viewport_state = PipelineViewportStateCreateInfo::builder();
    viewport_state.viewport_count = 1;
    viewport_state.scissor_count = 1;
    let rasterizer = PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(PolygonMode::FILL)
        .cull_mode(CullModeFlags::BACK)
        .front_face(FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisampling = PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(SampleCountFlagBits::_1);
    let blend_attachment = PipelineColorBlendAttachmentState {
        blend_enable: 0,
        color_write_mask: ColorComponentFlags::R
            | ColorComponentFlags::G
            | ColorComponentFlags::B
            | ColorComponentFlags::A,
        ..unsafe { core::mem::zeroed() }
    };
    let color_blending = PipelineColorBlendStateCreateInfo::builder()
        .attachments(std::slice::from_ref(&blend_attachment));
    let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];
    let dynamic_state = PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let pipeline_info = GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .p_vertex_input_state(&*vertex_input)
        .p_input_assembly_state(&*input_assembly)
        .p_viewport_state(&*viewport_state)
        .p_rasterization_state(&*rasterizer)
        .p_multisample_state(&*multisampling)
        .p_color_blend_state(&*color_blending)
        .p_dynamic_state(&*dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline =
        unsafe { device.create_graphics_pipelines(PipelineCache::null(), &[*pipeline_info], None) }
            .expect("Failed to create graphics pipeline")[0];

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    let framebuffers: Vec<Framebuffer> = image_views
        .iter()
        .map(|&view| {
            let views = [view];
            let fb_info = FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&views)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&fb_info, None) }
                .expect("Failed to create framebuffer")
        })
        .collect();

    // ════════════════════════════════════════════════════════════
    // Sync objects, render command buffer
    // ════════════════════════════════════════════════════════════
    let sem_info = SemaphoreCreateInfo::builder();
    let image_available =
        unsafe { device.create_semaphore(&sem_info, None) }.expect("Failed to create semaphore");
    let render_finished =
        unsafe { device.create_semaphore(&sem_info, None) }.expect("Failed to create semaphore");

    let fence_info = FenceCreateInfo::builder().flags(FenceCreateFlags::SIGNALED);
    let in_flight_fence =
        unsafe { device.create_fence(&fence_info, None) }.expect("Failed to create fence");

    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }
        .expect("Failed to allocate command buffer")[0];

    println!(
        "Texture example ready! Displaying {}x{} checkerboard.",
        width, height
    );

    VulkanState {
        window,
        instance,
        device,
        graphics_queue,
        swapchain,
        extent,
        render_pass,
        pipeline,
        pipeline_layout,
        framebuffers,
        image_views,
        command_pool,
        command_buffer,
        image_available,
        render_finished,
        in_flight_fence,
        surface,
        texture_image,
        texture_memory,
        texture_view,
        sampler,
        descriptor_set_layout,
        descriptor_pool,
        descriptor_set,
        staging_buffer,
        staging_memory,
    }
}

unsafe fn draw_frame(state: &VulkanState) {
    let d = &state.device;

    unsafe {
        d.wait_for_fences(&[state.in_flight_fence], true, u64::MAX)
            .expect("Failed to wait for fence");
        d.reset_fences(&[state.in_flight_fence])
            .expect("Failed to reset fence");

        let image_index = d
            .acquire_next_image_khr(
                state.swapchain,
                u64::MAX,
                state.image_available,
                Fence::null(),
            )
            .expect("Failed to acquire image");

        d.reset_command_buffer(state.command_buffer, CommandBufferResetFlags::empty())
            .expect("Failed to reset command buffer");

        record_commands(d, state, image_index);

        let wait_sems = [state.image_available];
        let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmd_bufs = [state.command_buffer];
        let signal_sems = [state.render_finished];
        let submit_info = SubmitInfo::builder()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_sems);

        d.queue_submit(state.graphics_queue, &[*submit_info], state.in_flight_fence)
            .expect("Failed to submit");

        let present_wait = [state.render_finished];
        let swapchains = [state.swapchain];
        let indices = [image_index];
        let present_info = PresentInfoKHR::builder()
            .wait_semaphores(&present_wait)
            .swapchains(&swapchains)
            .image_indices(&indices);

        let _ = d.queue_present_khr(state.graphics_queue, &present_info);
    }
}

unsafe fn record_commands(device: &Device, state: &VulkanState, image_index: u32) {
    unsafe {
        let begin_info = CommandBufferBeginInfo::builder();
        device
            .begin_command_buffer(state.command_buffer, &begin_info)
            .expect("Failed to begin command buffer");

        let clear_value = ClearValue {
            color: ClearColorValue {
                float32: [0.1, 0.1, 0.1, 1.0],
            },
        };
        let clear_values = [clear_value];

        let rp_begin = RenderPassBeginInfo::builder()
            .render_pass(state.render_pass)
            .framebuffer(state.framebuffers[image_index as usize])
            .render_area(Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: state.extent,
            })
            .clear_values(&clear_values);

        device.cmd_begin_render_pass(state.command_buffer, &rp_begin, SubpassContents::INLINE);

        device.cmd_bind_pipeline(
            state.command_buffer,
            PipelineBindPoint::GRAPHICS,
            state.pipeline,
        );

        // Bind the texture descriptor set
        device.cmd_bind_descriptor_sets(
            state.command_buffer,
            PipelineBindPoint::GRAPHICS,
            state.pipeline_layout,
            0,
            &[state.descriptor_set],
            &[],
        );

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: state.extent.width as f32,
            height: state.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(state.command_buffer, 0, &[viewport]);

        let scissor = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: state.extent,
        };
        device.cmd_set_scissor(state.command_buffer, 0, &[scissor]);

        // 6 vertices for a textured quad (two triangles)
        device.cmd_draw(state.command_buffer, 6, 1, 0, 0);

        device.cmd_end_render_pass(state.command_buffer);
        device
            .end_command_buffer(state.command_buffer)
            .expect("Failed to end command buffer");
    }
}

// Cleanup (textures.md cleanup section)
unsafe fn cleanup(state: VulkanState) {
    unsafe {
        state
            .device
            .device_wait_idle()
            .expect("Failed to wait idle");

        // Texture cleanup
        state.device.destroy_sampler(state.sampler, None);
        state.device.destroy_image_view(state.texture_view, None);
        state.device.destroy_image(state.texture_image, None);
        state.device.free_memory(state.texture_memory, None);
        state.device.destroy_buffer(state.staging_buffer, None);
        state.device.free_memory(state.staging_memory, None);

        // Descriptor cleanup
        state
            .device
            .destroy_descriptor_pool(state.descriptor_pool, None);
        state
            .device
            .destroy_descriptor_set_layout(state.descriptor_set_layout, None);

        // Standard cleanup
        state.device.destroy_fence(state.in_flight_fence, None);
        state.device.destroy_semaphore(state.render_finished, None);
        state.device.destroy_semaphore(state.image_available, None);
        state.device.destroy_command_pool(state.command_pool, None);
        for &fb in &state.framebuffers {
            state.device.destroy_framebuffer(fb, None);
        }
        state.device.destroy_pipeline(state.pipeline, None);
        state
            .device
            .destroy_pipeline_layout(state.pipeline_layout, None);
        state.device.destroy_render_pass(state.render_pass, None);
        for &v in &state.image_views {
            state.device.destroy_image_view(v, None);
        }
        state.device.destroy_swapchain_khr(state.swapchain, None);
        state.device.destroy_device(None);
        state.instance.destroy_surface(state.surface, None);
        state.instance.destroy_instance(None);
    }
}
