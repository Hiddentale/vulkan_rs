// Hello Triangle Part 2: Instance & Device + Swapchain & Surface
// Complete runnable program. Builds on Part 1.
// <https://hiddentale.github.io/vulkan_rust/getting-started/hello-triangle-2.html>

use vk::*;
use vulkan_rust::vk;
use vulkan_rust::{Entry, LibloadingLoader, Version};
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
    // ── Part 1: Load Vulkan ────────────────────────────────────
    let loader = LibloadingLoader::new().expect("Vulkan library not found");
    let entry = unsafe { Entry::new(loader) }.expect("Failed to load Vulkan");

    let version = entry.version().expect("Failed to query version");
    println!(
        "Vulkan {}.{}.{}",
        version.major, version.minor, version.patch
    );

    // ── Part 2: Create Instance with surface extensions ────────
    let surface_extensions = vulkan_rust::required_extensions();
    let extension_ptrs: Vec<*const i8> =
        surface_extensions.iter().map(|ext| ext.as_ptr()).collect();

    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    let layer_ptrs = [validation_layer.as_ptr()];

    let app_info = ApplicationInfo::builder()
        .p_application_name(c"Hello Triangle")
        .application_version(1)
        .p_engine_name(c"No Engine")
        .engine_version(1)
        .api_version(Version::new(1, 0, 0).to_raw());

    let create_info = InstanceCreateInfo::builder()
        .p_application_info(&app_info)
        .enabled_extension_names(&extension_ptrs)
        .enabled_layer_names(&layer_ptrs);

    let instance =
        unsafe { entry.create_instance(&create_info, None) }.expect("Failed to create instance");

    // ── Create Surface ─────────────────────────────────────────
    let surface =
        unsafe { instance.create_surface(window, window, None) }.expect("Failed to create surface");

    // ── Pick GPU with presentation support ─────────────────────
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
                    .unwrap_or(false);
            if graphics && present {
                physical_device = pd;
                graphics_family_index = i as u32;
                break 'outer;
            }
        }
    }
    assert!(!physical_device.is_null(), "No suitable GPU found");

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    let name: Vec<u8> = props
        .device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    println!("GPU: {}", String::from_utf8_lossy(&name));

    // ── Create Device with swapchain extension ─────────────────
    let device_extensions = [vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr()];
    let queue_priority = 1.0_f32;
    let queue_info = DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));
    let device_info = DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extensions);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }
        .expect("Failed to create device");
    let _queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

    // ── Query surface capabilities ─────────────────────────────
    let caps =
        unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }
            .expect("Failed to query surface capabilities");
    let formats =
        unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface) }
            .expect("Failed to query surface formats");
    let present_modes =
        unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface) }
            .expect("Failed to query present modes");

    // ── Choose swapchain settings ──────────────────────────────
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

    // ── Create swapchain ───────────────────────────────────────
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

    // ── Get swapchain images and create image views ────────────
    let images = unsafe { device.get_swapchain_images_khr(swapchain) }
        .expect("Failed to get swapchain images");
    println!(
        "Swapchain: {} images, {}x{}",
        images.len(),
        extent.width,
        extent.height
    );

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

    println!("Part 2 complete: {} image views created", image_views.len());

    // ── Clean up ───────────────────────────────────────────────
    unsafe {
        for &v in &image_views {
            device.destroy_image_view(v, None);
        }
        device.destroy_swapchain_khr(swapchain, None);
        device.destroy_device(None);
        instance.destroy_surface(surface, None);
        instance.destroy_instance(None);
    }
}
