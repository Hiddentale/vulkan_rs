# Handle Window Resize

> **Task:** Detect window resize events and recreate the swapchain
> without crashing or leaking resources.

## Prerequisites

- [Hello Triangle, Part 2](../getting-started/hello-triangle-2.md) (swapchain creation)
- [Synchronization](../concepts/synchronization.md) (device idle)
- [Implement Double Buffering](double-buffering.md) (frames in flight)

## The problem

When the window is resized, the swapchain images no longer match the
window dimensions. Vulkan tells you this has happened through two
mechanisms:

1. `acquire_next_image_khr` or `queue_present_khr` returns `ERROR_OUT_OF_DATE`,
   meaning the swapchain is no longer compatible with the surface.
2. `queue_present_khr` returns `SUBOPTIMAL`, meaning the swapchain still
   works but no longer matches the surface properties perfectly.

In either case, you must recreate the swapchain (and everything that
depends on its images) before rendering can continue.

## Step 1: Detect the resize

Track resize events from your windowing library and from Vulkan return
codes.

```rust,ignore
let mut framebuffer_resized = false;

// In your window event handler (winit example):
match event {
    WindowEvent::Resized(_) => {
        framebuffer_resized = true;
    }
    _ => {}
}
```

In the render loop, check both the flag and the Vulkan result codes:

```rust,ignore
use vulkan_rs::vk;
use vk::handles::*;
use vk::enums::Result as VkError;

let acquire_result = unsafe {
    device.acquire_next_image_khr(
        swapchain, u64::MAX, image_available_semaphore, Fence::null(),
    )
};

let image_index = match acquire_result {
    Ok(index) => index,
    Err(VkError::ERROR_OUT_OF_DATE) => {
        recreate_swapchain(/* ... */);
        continue; // restart this loop iteration
    }
    Err(e) => panic!("Failed to acquire swapchain image: {e:?}"),
};

// ... record and submit ...

let present_result = unsafe {
    device.queue_present_khr(graphics_queue, &present_info)
};

match present_result {
    Ok(_) => {}
    Err(VkError::ERROR_OUT_OF_DATE | VkError::SUBOPTIMAL) => {
        framebuffer_resized = false;
        recreate_swapchain(/* ... */);
    }
    Err(e) => panic!("Failed to present: {e:?}"),
}

// Also check the manual flag (some platforms don't always return OUT_OF_DATE).
if framebuffer_resized {
    framebuffer_resized = false;
    recreate_swapchain(/* ... */);
}
```

> *Before reading on: why do we check `framebuffer_resized` separately
> from the Vulkan error codes? Why not rely on `OUT_OF_DATE_KHR` alone?*

Some window systems (notably X11) do not always report out-of-date when
the window is resized. The manual flag from the window event handler
catches those cases.

## Step 2: Wait for the GPU

Before destroying any swapchain-related resources, all in-flight work
must finish.

```rust,ignore
unsafe { device.device_wait_idle() }
    .expect("Failed to wait for device idle");
```

This is simple and correct. For higher performance you could track
individual fences per swapchain image, but `device_wait_idle` is the
right choice for a resize path that runs infrequently.

## Step 3: Destroy old resources

Destroy everything that depends on the swapchain images, in reverse
creation order.

```rust,ignore
// Destroy framebuffers (one per swapchain image).
for &fb in &swapchain_framebuffers {
    unsafe { device.destroy_framebuffer(fb, None); }
}

// Destroy image views (one per swapchain image).
for &view in &swapchain_image_views {
    unsafe { device.destroy_image_view(view, None); }
}

// Do NOT destroy the old swapchain yet, we pass it to the new one.
```

You do not need to destroy the swapchain images themselves. They are
owned by the swapchain and will be cleaned up when the old swapchain is
destroyed.

## Step 4: Query new surface capabilities

The surface extent may have changed, so re-query it.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;

let surface_caps = unsafe {
    instance.get_physical_device_surface_capabilities_khr(physical_device, surface)
}
.expect("Failed to query surface capabilities");

let new_extent = if surface_caps.current_extent.width != u32::MAX {
    // The surface has a defined size, use it.
    surface_caps.current_extent
} else {
    // The surface size is undefined (e.g. Wayland), clamp to limits.
    let window_size = window.inner_size();
    Extent2D {
        width: window_size.width.clamp(
            surface_caps.min_image_extent.width,
            surface_caps.max_image_extent.width,
        ),
        height: window_size.height.clamp(
            surface_caps.min_image_extent.height,
            surface_caps.max_image_extent.height,
        ),
    }
};
```

## Step 5: Handle minimized windows

When a window is minimized, the surface extent can be `(0, 0)`. You
cannot create a swapchain with zero dimensions. Pause the render loop
until the window is restored.

```rust,ignore
if new_extent.width == 0 || new_extent.height == 0 {
    // Window is minimized. Wait for a resize event before continuing.
    // With winit, use Event::MainEventsCleared to avoid busy-waiting.
    return Ok(());
}
```

## Step 6: Create the new swapchain

Pass the old swapchain handle to `old_swapchain`. This lets the driver
reuse internal resources and can make the transition smoother.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

let old_swapchain = swapchain; // save the handle

let swapchain_info = SwapchainCreateInfoKHR::builder()
    .surface(surface)
    .min_image_count(desired_image_count)
    .image_format(surface_format.format)
    .image_color_space(surface_format.color_space)
    .image_extent(new_extent)
    .image_array_layers(1)
    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
    .image_sharing_mode(SharingMode::EXCLUSIVE)
    .pre_transform(surface_caps.current_transform)
    .composite_alpha(CompositeAlphaFlagBitsKHR::OPAQUE)
    .present_mode(present_mode)
    .clipped(1)
    .old_swapchain(old_swapchain); // <-- reuse hint

swapchain = unsafe { device.create_swapchain_khr(&swapchain_info, None) }
    .expect("Failed to create swapchain");

// Now destroy the old swapchain.
unsafe { device.destroy_swapchain_khr(old_swapchain, None); }
```

## Step 7: Recreate image views and framebuffers

The new swapchain has new images, so create fresh image views and
framebuffers.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain) }
    .expect("Failed to get swapchain images");

swapchain_image_views = swapchain_images
    .iter()
    .map(|&image| {
        let view_info = ImageViewCreateInfo::builder()
            .image(image)
            .view_type(ImageViewType::_2D)
            .format(surface_format.format)
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

swapchain_framebuffers = swapchain_image_views
    .iter()
    .map(|&view| {
        let attachments = [view];
        let fb_info = FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(new_extent.width)
            .height(new_extent.height)
            .layers(1);
        unsafe { device.create_framebuffer(&fb_info, None) }
            .expect("Failed to create framebuffer")
    })
    .collect();
```

## Putting it all together

A helper function that bundles the recreation logic:

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::handles::*;

fn recreate_swapchain(
    instance: &vulkan_rs::Instance,
    device: &vulkan_rs::Device,
    physical_device: PhysicalDevice,
    surface: SurfaceKHR,
    window: &winit::window::Window,
    render_pass: RenderPass,
    swapchain: &mut SwapchainKHR,
    swapchain_image_views: &mut Vec<ImageView>,
    swapchain_framebuffers: &mut Vec<Framebuffer>,
    surface_format: SurfaceFormatKHR,
    present_mode: PresentModeKHR,
) -> Extent2D {
    unsafe {
        device.device_wait_idle()
            .expect("Failed to wait for device idle");

        // Destroy old framebuffers and image views.
        for &fb in swapchain_framebuffers.iter() {
            device.destroy_framebuffer(fb, None);
        }
        for &view in swapchain_image_views.iter() {
            device.destroy_image_view(view, None);
        }
    }

    // Query new extent, create new swapchain, views, framebuffers.
    // ... (Steps 4 through 7 from above) ...

    new_extent
}
```

## Common mistakes

**Forgetting to update the viewport and scissor.** If you use dynamic
viewport/scissor state (which you should), update them to the new extent
each frame. If you baked them into the pipeline, you need to recreate
the pipeline too.

**Leaking old image views.** Every `create_image_view` must have a
matching `destroy_image_view`. If you overwrite the `Vec` without
destroying the old views first, those handles leak.

**Not handling SUBOPTIMAL.** `SUBOPTIMAL` from `queue_present_khr` is
not a fatal error, but ignoring it means you render at the wrong
resolution until something else triggers an `ERROR_OUT_OF_DATE`.

## Notes

- **Depth buffers.** If your render pass uses a depth attachment, you
  must also recreate the depth image, its memory, and its image view
  when the swapchain extent changes.
- **Render pass compatibility.** The render pass itself does not depend
  on the swapchain extent, only on the image format. You do not need to
  recreate it unless the surface format changes (which is extremely rare).
- **Dynamic state.** Using `DynamicState::VIEWPORT` and
  `DynamicState::SCISSOR` avoids having to recreate the pipeline on
  resize. This is the recommended approach.
