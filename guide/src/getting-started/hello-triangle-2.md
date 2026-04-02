# Hello Triangle, Part 2: Swapchain & Surface

In [Part 1](hello-triangle-1.md) we loaded Vulkan, created an Instance
and Device, and retrieved a graphics queue. We can talk to the GPU, but
we have nowhere to *show* anything.

**What we build in this part:**

```text
Open a window ──> Create Surface ──> Create Swapchain ──> Get image views
                                                          + validation layers
```

By the end of this part, we will have a window with a swapchain ready
to receive rendered frames.

## New dependencies

We need a windowing library. This tutorial uses `winit`, but `vulkan_rs`
works with anything that implements `raw-window-handle`.

```toml
[dependencies]
vulkan-rs = "0.1"
winit = "0.30"
```

## Step 1: Open a window

Before creating a Vulkan surface, we need a platform window.

```rust,ignore
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

        // ... Vulkan initialization uses &window here ...

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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
```

## Step 2: Create the Instance with surface extensions

In Part 1 we created an Instance with no extensions. Now we need the
platform surface extensions so Vulkan can render to our window.

`vulkan_rs` provides `required_extensions()` which returns the right
extensions for your platform.

```rust,ignore
use vulkan_rs::{Entry, LibloadingLoader};
use vulkan_rs::vk;
use vk::structs::*;

// ── Load Vulkan ────────────────────────────────────────────────
let loader = LibloadingLoader::new()
    .expect("Vulkan library not found");
let entry = unsafe { Entry::new(loader) }
    .expect("Failed to load Vulkan");

// ── Gather required extensions ─────────────────────────────────
//
// required_extensions() returns platform-specific extensions:
//   Windows: VK_KHR_surface + VK_KHR_win32_surface
//   Linux:   VK_KHR_surface + VK_KHR_xlib_surface + VK_KHR_wayland_surface
//   macOS:   VK_KHR_surface + VK_EXT_metal_surface
let surface_extensions = vulkan_rs::required_extensions();
let extension_ptrs: Vec<*const i8> = surface_extensions
    .iter()
    .map(|ext| ext.as_ptr())
    .collect();

// ── Enable the validation layer ────────────────────────────────
//
// Always enable during development. See the Validation Layers
// concept chapter for details.
let validation_layer = c"VK_LAYER_KHRONOS_validation";
let layer_ptrs = [validation_layer.as_ptr()];

// ── Create the instance ────────────────────────────────────────
let app_info = ApplicationInfo::builder()
    .p_application_name(c"Hello Triangle".as_ptr())
    .application_version(1)
    .p_engine_name(c"No Engine".as_ptr())
    .engine_version(1)
    .api_version(1 << 22);  // Vulkan 1.0

let create_info = InstanceCreateInfo::builder()
    .p_application_info(&*app_info)
    .enabled_extension_names(&extension_ptrs)
    .enabled_layer_names(&layer_ptrs);

let instance = unsafe { entry.create_instance(&create_info, None) }
    .expect("Failed to create instance");
```

> *Before reading on: we enabled validation layers here but did not set
> up a debug messenger callback. What happens to validation errors?*
>
> They go to stderr on most platforms. Setting up a debug messenger
> (as shown in the [Validation chapter](../concepts/validation.md))
> gives you programmatic control over the output. For a tutorial,
> stderr is fine.

## Step 3: Create a Surface

A Surface is Vulkan's abstraction over a platform window. It represents
the thing you render to: a Win32 HWND, an X11 Window, a Wayland
wl_surface, etc.

`vulkan_rs` provides `instance.create_surface()` which handles the
platform dispatch for you via `raw-window-handle`.

```rust,ignore
// ── Create the surface ─────────────────────────────────────────
//
// create_surface uses raw-window-handle to detect the platform
// and call the right vkCreate*Surface function.
let surface = unsafe { instance.create_surface(&window, &window, None) }
    .expect("Failed to create surface");
```

The surface is an Instance-level object. It must be destroyed before
the Instance.

## Step 4: Pick a GPU (with presentation support)

In Part 1 we picked the first GPU. Now we also need to verify it can
**present** to our surface, which means it has a queue family that
supports both graphics and presentation.

```rust,ignore
// ── Enumerate GPUs ─────────────────────────────────────────────
let physical_devices = unsafe { instance.enumerate_physical_devices() }
    .expect("Failed to enumerate GPUs");

// ── Find a GPU with a queue family that supports both graphics
//    and presentation to our surface ────────────────────────────
use vk::handles::*;

let mut physical_device = PhysicalDevice::null();
let mut graphics_family_index = 0u32;

'outer: for &pd in &physical_devices {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(pd)
    };

    for (i, family) in queue_families.iter().enumerate() {
        let supports_graphics =
            family.queue_flags & QueueFlags::GRAPHICS
            != QueueFlags::empty();

        // Check if this queue family can present to our surface.
        let supports_present = unsafe {
            instance.get_physical_device_surface_support_khr(
                pd,
                i as u32,
                surface,
            )
        }
        .unwrap_or(0) != 0;

        if supports_graphics && supports_present {
            physical_device = pd;
            graphics_family_index = i as u32;
            break 'outer;
        }
    }
}

assert!(
    !physical_device.is_null(),
    "No GPU found with graphics + presentation support"
);
```

> *Before reading on: why do we check for presentation support
> separately from graphics support? Can a queue family support
> graphics but not presentation?*
>
> Yes. On some hardware, a queue family can execute graphics commands
> but cannot present to a specific surface. Presentation support
> depends on both the queue family *and* the surface (which is tied
> to a specific monitor/display). Always check with
> `get_physical_device_surface_support_khr`.

## Step 5: Create the Device with the swapchain extension

Now we add `VK_KHR_swapchain`, the extension that lets us create a
swapchain.

```rust,ignore
use vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME;
let device_extensions = [KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr()];

let queue_priority = 1.0_f32;
let queue_info = DeviceQueueCreateInfo::builder()
    .queue_family_index(graphics_family_index)
    .queue_priorities(std::slice::from_ref(&queue_priority));

let device_info = DeviceCreateInfo::builder()
    .queue_create_infos(std::slice::from_ref(&*queue_info))
    .enabled_extension_names(&device_extensions);

let device = unsafe {
    instance.create_device(physical_device, &device_info, None)
}
.expect("Failed to create device");

let graphics_queue = unsafe {
    device.get_device_queue(graphics_family_index, 0)
};
```

## Step 6: Query surface capabilities

Before creating a swapchain, we must ask the surface what it supports:
image formats, present modes, minimum/maximum image count, and
supported image sizes.

```rust,ignore
// ── Query what the surface supports ────────────────────────────
let capabilities = unsafe {
    instance.get_physical_device_surface_capabilities_khr(
        physical_device,
        surface,
    )
}
.expect("Failed to query surface capabilities");

let formats = unsafe {
    instance.get_physical_device_surface_formats_khr(
        physical_device,
        surface,
    )
}
.expect("Failed to query surface formats");

let present_modes = unsafe {
    instance.get_physical_device_surface_present_modes_khr(
        physical_device,
        surface,
    )
}
.expect("Failed to query present modes");
```

## Step 7: Choose swapchain settings

We need to decide three things: the image format, the present mode,
and the image extent (resolution).

```rust,ignore
use vk::enums::*;
use vk::bitmasks::*;

// ── Choose format ──────────────────────────────────────────────
//
// Prefer B8G8R8A8_SRGB with SRGB_NONLINEAR color space.
// Fall back to whatever is available.
let surface_format = formats
    .iter()
    .find(|f| {
        f.format == Format::B8G8R8A8_SRGB
            && f.color_space == ColorSpaceKHR::SRGB_NONLINEAR
    })
    .unwrap_or(&formats[0]);

// ── Choose present mode ────────────────────────────────────────
//
// MAILBOX = triple buffering (low latency, no tearing).
// FIFO = vsync (guaranteed available).
let present_mode = if present_modes.contains(&PresentModeKHR::MAILBOX) {
    PresentModeKHR::MAILBOX
} else {
    PresentModeKHR::FIFO  // always available
};

// ── Choose extent (resolution) ─────────────────────────────────
//
// If current_extent is 0xFFFFFFFF, the surface size is determined
// by the swapchain extent. Otherwise, use the surface's size.
let extent = if capabilities.current_extent.width != u32::MAX {
    capabilities.current_extent
} else {
    let size = window.inner_size();
    Extent2D {
        width: size.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: size.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
};

// ── Choose image count ─────────────────────────────────────────
//
// Request one more than the minimum so we always have an image
// to render to while the display is reading another.
let image_count = {
    let desired = capabilities.min_image_count + 1;
    if capabilities.max_image_count > 0 {
        desired.min(capabilities.max_image_count)
    } else {
        desired // max_image_count == 0 means no upper limit
    }
};
```

## Step 8: Create the swapchain

```rust,ignore
let swapchain_info = SwapchainCreateInfoKHR::builder()
    .surface(surface)
    .min_image_count(image_count)
    .image_format(surface_format.format)
    .image_color_space(surface_format.color_space)
    .image_extent(extent)
    .image_array_layers(1)
    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
    .image_sharing_mode(SharingMode::EXCLUSIVE)
    .pre_transform(capabilities.current_transform)
    .composite_alpha(CompositeAlphaFlagBitsKHR::OPAQUE)
    .present_mode(present_mode)
    .clipped(true)       // discard pixels behind other windows
    .old_swapchain(SwapchainKHR::null());

let swapchain = unsafe {
    device.create_swapchain_khr(&swapchain_info, None)
}
.expect("Failed to create swapchain");
```

The swapchain now owns a set of images. We retrieve their handles next.

## Step 9: Get swapchain images and create image views

The swapchain images are owned by the swapchain, so we do not destroy
them ourselves. But we need **image views** to use them in render passes
and framebuffers.

```rust,ignore
// ── Get the swapchain images ───────────────────────────────────
let swapchain_images = unsafe {
    device.get_swapchain_images_khr(swapchain)
}
.expect("Failed to get swapchain images");

println!("Swapchain has {} images", swapchain_images.len());

// ── Create an image view for each swapchain image ──────────────
let swapchain_image_views: Vec<ImageView> = swapchain_images
    .iter()
    .map(|&image| {
        let view_info = ImageViewCreateInfo::builder()
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

        unsafe { device.create_image_view(&view_info, None) }
            .expect("Failed to create image view")
    })
    .collect();
```

## Where we are now

At this point we have:

```text
Window (winit)
  │
  └── Surface (VK_KHR_surface)
        │
        └── Swapchain (VK_KHR_swapchain)
              │
              ├── Image 0 ──> ImageView 0
              ├── Image 1 ──> ImageView 1
              └── Image 2 ──> ImageView 2
```

The swapchain gives us images to render into. The image views let us
use those images in render passes. In Part 3, we will create a render
pass and a graphics pipeline so we can actually draw something.

## Clean up

Destruction in reverse creation order:

```rust,ignore
unsafe {
    // Image views (we created these)
    for &view in &swapchain_image_views {
        device.destroy_image_view(view, None);
    }

    // Swapchain (device-level)
    device.destroy_swapchain_khr(swapchain, None);

    // Device
    device.destroy_device(None);

    // Surface (instance-level, before instance)
    instance.destroy_surface(surface, None);

    // Instance
    instance.destroy_instance(None);
}
```

## What we learned

| Step | What | Why |
|------|------|-----|
| Surface extensions | `required_extensions()` | Platform-specific window integration |
| Validation layer | `VK_LAYER_KHRONOS_validation` | Catch mistakes during development |
| Surface | `instance.create_surface()` | Connect Vulkan to a window |
| Presentation check | `get_physical_device_surface_support_khr` | Ensure the GPU can present to this surface |
| Swapchain extension | `VK_KHR_swapchain` | Enable swapchain creation on the device |
| Surface capabilities | `get_physical_device_surface_capabilities_khr` | Query supported formats, sizes, present modes |
| Swapchain | `create_swapchain_khr` | A set of images the display rotates through |
| Image views | `create_image_view` | Make swapchain images usable by render passes |

## Concepts to explore

- [Validation Layers & Debugging](../concepts/validation.md), how to
  set up a debug messenger for better error output.
- [The Vulkan Object Model](../concepts/object-model.md), why we destroy
  in reverse order.

## Exercises

1. **Print all surface formats.** Before choosing a format, print every
   format and color space the surface supports.
2. **Print the chosen present mode.** Print which present mode was
   selected (MAILBOX or FIFO) and why.
3. **Handle no validation layer.** What happens if the validation layer
   is not installed? Modify the code to check for its availability with
   `enumerate_instance_layer_properties` and skip it gracefully.

## Next

[Part 3: Render Pass & Pipeline](hello-triangle-3.md) creates the
graphics pipeline that defines *how* we draw our triangle.
