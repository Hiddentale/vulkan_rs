# Hello Triangle, Part 4: Command Buffers & Drawing

This is the final part. In [Part 3](hello-triangle-3.md) we created
the render pass, pipeline, and framebuffers. Now we record commands,
submit them, and present a triangle to the screen.

**What we build in this part:**

```text
Create sync objects ──> Create command pool/buffers
       │                        │
       └──> Render loop: acquire image ──> record commands ──> submit ──> present
```

This part ties together every concept from the previous three parts.
When you see the triangle, you will have written a complete Vulkan
application.

## Step 1: Create synchronization objects

We need fences and semaphores to coordinate CPU and GPU work. See
[Synchronization](../concepts/synchronization.md) for the full concept.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// ── Semaphores: GPU-to-GPU synchronization ─────────────────────
let sem_info = SemaphoreCreateInfo::builder();

// "The swapchain image is ready to render into."
let image_available = unsafe { device.create_semaphore(&sem_info, None) }
    .expect("Failed to create semaphore");

// "Rendering is done, safe to present."
let render_finished = unsafe { device.create_semaphore(&sem_info, None) }
    .expect("Failed to create semaphore");

// ── Fence: GPU-to-CPU synchronization ──────────────────────────
//
// SIGNALED so the first frame doesn't block forever waiting for
// a "previous frame" that never existed.
let fence_info = FenceCreateInfo::builder()
    .flags(FenceCreateFlags::SIGNALED);

let in_flight_fence = unsafe { device.create_fence(&fence_info, None) }
    .expect("Failed to create fence");
```

> *Before reading on: why do we create the fence with SIGNALED? What
> would happen on the first frame if we didn't?*
>
> The render loop starts by waiting for the fence. On the first frame,
> no GPU work has been submitted yet, so an unsignaled fence would block
> forever. Starting it signaled lets the first frame pass through
> immediately.

## Step 2: Create a command pool and command buffer

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// ── Command pool ───────────────────────────────────────────────
let pool_info = CommandPoolCreateInfo::builder()
    .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
    .queue_family_index(graphics_family_index);

let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
    .expect("Failed to create command pool");

// ── Allocate one command buffer ────────────────────────────────
let alloc_info = CommandBufferAllocateInfo::builder()
    .command_pool(command_pool)
    .level(CommandBufferLevel::PRIMARY)
    .command_buffer_count(1);

let command_buffer = unsafe {
    device.allocate_command_buffers(&alloc_info)
}
.expect("Failed to allocate command buffer")[0];
```

## Step 3: Record drawing commands

This function records all the commands needed to draw one frame. We
call it every frame with the correct framebuffer for the current
swapchain image.

```rust,ignore
use vulkan_rust::vk;
use vk::*;

unsafe fn record_commands(
    device: &vulkan_rust::Device,
    command_buffer: CommandBuffer,
    render_pass: RenderPass,
    framebuffer: Framebuffer,
    pipeline: Pipeline,
    extent: Extent2D,
) {
    unsafe {
    // ── Begin recording ────────────────────────────────────────
    let begin_info = CommandBufferBeginInfo::builder();
    device.begin_command_buffer(command_buffer, &begin_info)
        .expect("Failed to begin command buffer");

    // ── Begin render pass ──────────────────────────────────────
    let clear_value = ClearValue {
        color: ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],  // black
        },
    };

    let clear_values = [clear_value];
    let rp_begin = RenderPassBeginInfo::builder()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent,
        })
        .clear_values(&clear_values);

    device.cmd_begin_render_pass(
        command_buffer,
        &rp_begin,
        SubpassContents::INLINE,
    );

    // ── Bind the pipeline ──────────────────────────────────────
    device.cmd_bind_pipeline(
        command_buffer,
        PipelineBindPoint::GRAPHICS,
        pipeline,
    );

    // ── Set dynamic viewport and scissor ───────────────────────
    let viewport = Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    device.cmd_set_viewport(command_buffer, 0, &[viewport]);

    let scissor = Rect2D {
        offset: Offset2D { x: 0, y: 0 },
        extent,
    };
    device.cmd_set_scissor(command_buffer, 0, &[scissor]);

    // ── Draw the triangle ──────────────────────────────────────
    //
    // 3 vertices, 1 instance, starting at vertex 0, instance 0.
    // The vertex data is hard-coded in the shader.
    device.cmd_draw(command_buffer, 3, 1, 0, 0);

    // ── End render pass and recording ──────────────────────────
    device.cmd_end_render_pass(command_buffer);
    device.end_command_buffer(command_buffer)
        .expect("Failed to end command buffer");
    }
}
```

This is the core of every Vulkan frame: begin recording, begin render
pass, bind pipeline, set state, draw, end render pass, end recording.

## Step 4: The render loop

Now we tie everything together in the event loop. Each frame:

1. **Wait** for the previous frame's fence (CPU waits for GPU).
2. **Acquire** the next swapchain image (GPU signals `image_available`).
3. **Record** commands into the command buffer.
4. **Submit** the command buffer (waits on `image_available`, signals
   `render_finished` and the fence).
5. **Present** the image (waits on `render_finished`).

```rust,ignore
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;

impl ApplicationHandler for App {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        // Window and Vulkan setup already done (see Part 2).
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                unsafe { self.draw_frame() };
                // Request the next frame immediately.
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

// In main:
let event_loop = EventLoop::new().expect("Failed to create event loop");
event_loop.run_app(&mut app).expect("Event loop error");
```

The `draw_frame` function:

```rust,ignore
use vulkan_rust::vk;
use vk::*;

unsafe fn draw_frame(
    device: &vulkan_rust::Device,
    swapchain: SwapchainKHR,
    in_flight_fence: Fence,
    image_available: Semaphore,
    render_finished: Semaphore,
    command_buffer: CommandBuffer,
    framebuffers: &[Framebuffer],
    render_pass: RenderPass,
    pipeline: Pipeline,
    extent: Extent2D,
    graphics_queue: Queue,
) {
    unsafe {
    // ── 1. Wait for previous frame ─────────────────────────────
    device.wait_for_fences(&[in_flight_fence], true, u64::MAX)
        .expect("Failed to wait for fence");
    device.reset_fences(&[in_flight_fence])
        .expect("Failed to reset fence");

    // ── 2. Acquire next swapchain image ────────────────────────
    let image_index = device
        .acquire_next_image_khr(
            swapchain,
            u64::MAX,
            image_available,
            Fence::null(),
        )
        .expect("Failed to acquire swapchain image");

    // ── 3. Record commands ─────────────────────────────────────
    device.reset_command_buffer(
        command_buffer,
        CommandBufferResetFlags::empty(),
    )
    .expect("Failed to reset command buffer");

    record_commands(
        device,
        command_buffer,
        render_pass,
        framebuffers[image_index as usize],
        pipeline,
        extent,
    );

    // ── 4. Submit ──────────────────────────────────────────────
    let wait_sems = [image_available];
    let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let cmd_bufs = [command_buffer];
    let signal_sems = [render_finished];

    let submit_info = SubmitInfo::builder()
        .wait_semaphores(&wait_sems)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&cmd_bufs)
        .signal_semaphores(&signal_sems);

    device.queue_submit(graphics_queue, &[*submit_info], in_flight_fence)
        .expect("Failed to submit draw command buffer");

    // ── 5. Present ─────────────────────────────────────────────
    let present_wait = [render_finished];
    let swapchains = [swapchain];
    let indices = [image_index];
    let present_info = PresentInfoKHR::builder()
        .wait_semaphores(&present_wait)
        .swapchains(&swapchains)
        .image_indices(&indices);

    device.queue_present_khr(graphics_queue, &present_info)
        .expect("Failed to present");
    }
}
```

The synchronization flow each frame:

```text
CPU: wait_for_fences ────────────────────────────────────> (free to continue)
       │
       v
GPU: acquire_next_image ──signals──> image_available
                                        │ (GPU waits at COLOR_ATTACHMENT_OUTPUT)
                                        v 
GPU:                        queue_submit ──signals──> render_finished
                                                        │ ──signals──> in_flight_fence
                                                        v                    │
GPU:                                             queue_present               │
                                                                             │ 
CPU: (next frame)                                   wait_for_fences <────────┘
```

## Step 5: Wait before cleanup

Before destroying anything, wait for the GPU to finish all work:

```rust,ignore
// After the event loop exits:
unsafe { device.device_wait_idle() }
    .expect("Failed to wait for device idle");
```

Then destroy everything in reverse creation order:

```rust,ignore
unsafe {
    device.destroy_fence(in_flight_fence, None);
    device.destroy_semaphore(render_finished, None);
    device.destroy_semaphore(image_available, None);
    device.destroy_command_pool(command_pool, None);

    for &fb in &framebuffers {
        device.destroy_framebuffer(fb, None);
    }
    device.destroy_pipeline(pipeline, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
    device.destroy_render_pass(render_pass, None);

    for &view in &swapchain_image_views {
        device.destroy_image_view(view, None);
    }
    device.destroy_swapchain_khr(swapchain, None);

    device.destroy_device(None);
    instance.destroy_surface(surface, None);
    instance.destroy_instance(None);
}
```

## You did it

Run `cargo run`. You should see a window with a colored triangle on a
black background:

```text
┌──────────────────────────────────────┐
│                                      │
│              ▲ (red)                 │
│             ╱ ╲                      │
│            ╱   ╲                     │
│   (blue)  ╱     ╲  (green)           │
│          ▔▔▔▔▔                   │
│                                      │
└──────────────────────────────────────┘
```

If you see a black window with no triangle, check these common issues:

1. **Validation errors in the console.** Read them. They usually point
   directly at the problem.
2. **Front face winding.** If your triangle vertices are wound
   counter-clockwise but you set `CLOCKWISE`, the triangle is culled.
   Try `CullModeFlags::NONE` to test.
3. **Missing SPIR-V files.** `include_bytes!` panics at compile time
   if the file is not found.

## What we built across all four parts

```text
Part 1: Entry ──> Instance ──> PhysicalDevice ──> Device ──> Queue
Part 2: Window ──> Surface ──> Swapchain ──> ImageViews
Part 3: Shaders ──> RenderPass ──> Pipeline ──> Framebuffers
Part 4: Sync objects ──> CommandPool/Buffer ──> Render loop
```

Every Vulkan application follows this structure. The details change
(more pipelines, more buffers, more complex synchronization), but
the architecture is the same.

## What we skipped

This tutorial focused on getting a triangle on screen. A production
application would add:

- **Multiple frames in flight** to avoid the CPU waiting for the GPU
  every frame. See [Double Buffering](../how-to/double-buffering.md).
- **Window resize handling** to recreate the swapchain when the window
  size changes. See [Handle Window Resize](../how-to/resize.md).
- **Vertex buffers** to pass vertex data from CPU memory to the GPU.
  See [Memory Management](../concepts/memory.md).
- **Descriptor sets** to pass uniforms and textures to shaders.
  See [Descriptor Sets](../concepts/descriptors.md).
- **Depth testing** for 3D rendering.

## Exercises

1. **Change the triangle color.** Modify the fragment shader (or the
   vertex shader's color array) and recompile the SPIR-V.
2. **Draw a rectangle.** Change the shader to output 6 vertices (two
   triangles) and update the `cmd_draw` vertex count.
3. **Add frames in flight.** Create two sets of sync objects and command
   buffers. Alternate between them each frame so the CPU can record
   frame N+1 while the GPU renders frame N.
4. **Handle resize.** When the window is resized, recreate the
   swapchain, image views, and framebuffers. The
   [Handle Window Resize](../how-to/resize.md) guide covers this.

## Where to go from here

- **[Concepts section](../concepts/how-to-read.md)**: deep dives into
  every Vulkan subsystem.
- **[How-To Guides](../how-to/textures.md)**: recipes for specific
  tasks (textures, resize, push constants).
- **[API reference](https://docs.rs/vulkan-rust)**: every type and method
  with spec links and error codes.
