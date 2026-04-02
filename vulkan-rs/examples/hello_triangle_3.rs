// Hello Triangle Part 3: Parts 1+2 setup + Render Pass & Pipeline
// Complete runnable program.
// <https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-3.html>

use vk::bitmasks::*;
use vk::enums::*;
use vk::handles::*;
use vk::structs::*;
use vulkan_rs::vk;
use vulkan_rs::{Entry, LibloadingLoader, cast_to_u32};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Hello Triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        run(&window);

        self.window = Some(window);
        event_loop.exit();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App { window: None };
    event_loop.run_app(&mut app).expect("Event loop error");
}

fn run(window: &Window) {
    // ════════════════════════════════════════════════════════════
    // Part 1: Load Vulkan
    // ════════════════════════════════════════════════════════════
    let loader = LibloadingLoader::new().expect("Vulkan library not found");
    let entry = unsafe { Entry::new(loader) }.expect("Failed to load Vulkan");

    let version = entry.version().expect("Failed to query version");
    println!(
        "Vulkan {}.{}.{}",
        version.major, version.minor, version.patch
    );

    // ════════════════════════════════════════════════════════════
    // Part 2: Instance, Surface, Device, Swapchain
    // ════════════════════════════════════════════════════════════
    let surface_extensions = vulkan_rs::required_extensions();
    let extension_ptrs: Vec<*const i8> =
        surface_extensions.iter().map(|ext| ext.as_ptr()).collect();

    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    let layer_ptrs = [validation_layer.as_ptr()];

    let app_info = ApplicationInfo::builder()
        .p_application_name(c"Hello Triangle".as_ptr())
        .application_version(1)
        .p_engine_name(c"No Engine".as_ptr())
        .engine_version(1)
        .api_version(1 << 22);

    let create_info = InstanceCreateInfo::builder()
        .p_application_info(&*app_info)
        .enabled_extension_names(&extension_ptrs)
        .enabled_layer_names(&layer_ptrs);

    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");

    let surface =
        unsafe { instance.create_surface(window, window, None) }.expect("Failed to create surface");

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

    let device_extensions = [vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr()];
    let queue_priority = 1.0_f32;
    let queue_info = DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));
    let device_info = DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&*queue_info))
        .enabled_extension_names(&device_extensions);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .expect("Failed to create device");
    let _queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

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

    println!(
        "Swapchain: {} images, {}x{}",
        images.len(),
        extent.width,
        extent.height
    );

    // ════════════════════════════════════════════════════════════
    // Part 3: Shaders, Render Pass, Pipeline, Framebuffers
    // ════════════════════════════════════════════════════════════

    // ── Load shaders ───────────────────────────────────────────
    let vert_bytes = include_bytes!("shaders/triangle.vert.spv");
    let frag_bytes = include_bytes!("shaders/triangle.frag.spv");

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

    // ── Create render pass ─────────────────────────────────────
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

    // ── Create pipeline layout ─────────────────────────────────
    let layout_info = PipelineLayoutCreateInfo::builder();
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .expect("Failed to create pipeline layout");

    // ── Create graphics pipeline ───────────────────────────────
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

    // ── Create framebuffers ────────────────────────────────────
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

    println!(
        "Part 3 complete: pipeline + {} framebuffers created",
        framebuffers.len()
    );

    // ── Clean up ───────────────────────────────────────────────
    unsafe {
        for &fb in &framebuffers {
            device.destroy_framebuffer(fb, None);
        }
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);
        for &v in &image_views {
            device.destroy_image_view(v, None);
        }
        device.destroy_swapchain_khr(swapchain, None);
        device.destroy_device(None);
        instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
