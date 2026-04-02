# Implement Double Buffering

> **Task:** Set up frames-in-flight so the CPU records frame N+1 while
> the GPU renders frame N.

## Prerequisites

- [Synchronization](../concepts/synchronization.md) (fences, semaphores)
- [Command Buffers](../concepts/command-buffers.md)
- [Hello Triangle, Part 4](../getting-started/hello-triangle-4.md) (basic render loop)

## The problem

In a single-buffered render loop, the CPU submits a frame and then waits
for the GPU to finish before it can start recording the next frame. This
means the CPU sits idle during GPU rendering, and the GPU sits idle
during CPU recording. You get roughly half the throughput you could.

```text
Single buffered:
CPU: [record 0]...........[record 1]...........[record 2]...
GPU: ...........[render 0]...........[render 1]...........[render 2]
     └── idle ──┘         └── idle ──┘
```

With double buffering (two frames in flight), the CPU records the next
frame while the GPU is still rendering the current one:

```text
Double buffered:
CPU: [record 0][record 1][record 2][record 3]...
GPU: .......[render 0][render 1][render 2][render 3]...
```

The overlap keeps both processors busy.

## Step 1: Define the frame count

Two frames in flight is the standard choice. Three is occasionally used,
but adds latency without much throughput gain on most hardware.

```rust,ignore
const MAX_FRAMES_IN_FLIGHT: usize = 2;
```

## Step 2: Create per-frame synchronization objects

Each frame in flight needs its own set of sync primitives:

- **Fence:** the CPU waits on this before reusing the frame's resources.
- **Image-available semaphore:** signals when the swapchain image is
  ready to be rendered into.
- **Render-finished semaphore:** signals when rendering is done and the
  image can be presented.

> *Before reading on: why does each frame need its own fence? What would
> go wrong if all frames shared a single fence?*

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::handles::*;

struct FrameSync {
    in_flight_fence: Fence,
    image_available: Semaphore,
    render_finished: Semaphore,
}

let fence_info = FenceCreateInfo::builder()
    .flags(FenceCreateFlags::SIGNALED); // start signaled so frame 0 doesn't deadlock

let semaphore_info = SemaphoreCreateInfo::builder();

let mut frame_sync = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
for _ in 0..MAX_FRAMES_IN_FLIGHT {
    let sync = FrameSync {
        in_flight_fence: unsafe { device.create_fence(&fence_info, None) }
            .expect("Failed to create fence"),
        image_available: unsafe { device.create_semaphore(&semaphore_info, None) }
            .expect("Failed to create semaphore"),
        render_finished: unsafe { device.create_semaphore(&semaphore_info, None) }
            .expect("Failed to create semaphore"),
    };
    frame_sync.push(sync);
}
```

Note the `SIGNALED` flag on the fences. The render loop starts by
waiting on the fence, so frame 0 needs the fence to be signaled already
or the first `wait_for_fences` call will block forever.

## Step 3: Create per-frame command buffers

Each frame in flight needs its own command buffer so the CPU can record
into one while the GPU executes the other.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::handles::*;

let alloc_info = CommandBufferAllocateInfo::builder()
    .command_pool(command_pool)
    .level(CommandBufferLevel::PRIMARY)
    .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

let mut command_buffers = vec![CommandBuffer::null(); MAX_FRAMES_IN_FLIGHT];
unsafe {
    device.allocate_command_buffers(&alloc_info, command_buffers.as_mut_ptr())
}
.expect("Failed to allocate command buffers");
```

## Step 4: The render loop

The frame index cycles through `0..MAX_FRAMES_IN_FLIGHT`. Each
iteration uses only the resources belonging to that frame index.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::handles::*;

let mut current_frame: usize = 0;

loop {
    // Handle window events (poll_events, etc.)
    // ...

    let sync = &frame_sync[current_frame];
    let cmd = command_buffers[current_frame];

    unsafe {
    // --- 1. Wait for this frame's previous submission to finish ---
    device.wait_for_fences(&[sync.in_flight_fence], 1, u64::MAX)
        .expect("Failed to wait for fence");

    // --- 2. Acquire the next swapchain image ---
    let image_index = device.acquire_next_image_khr(
        swapchain,
        u64::MAX,
        sync.image_available,  // signaled when image is ready
        Fence::null(),
    )
    .expect("Failed to acquire swapchain image");

    // --- 3. Reset the fence only after we know we will submit work ---
    // Resetting before acquire_next_image could deadlock if acquire fails.
    device.reset_fences(&[sync.in_flight_fence])
        .expect("Failed to reset fence");

    // --- 4. Record commands ---
    device.reset_command_buffer(cmd, CommandBufferResetFlags::empty())
        .expect("Failed to reset command buffer");

    let begin_info = CommandBufferBeginInfo::builder();
    device.begin_command_buffer(cmd, &begin_info)
        .expect("Failed to begin command buffer");

    // ... record render pass, draw calls, etc. ...

    device.end_command_buffer(cmd)
        .expect("Failed to end command buffer");

    // --- 5. Submit ---
    let wait_semaphores = [sync.image_available];
    let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let signal_semaphores = [sync.render_finished];
    let command_buffers_to_submit = [cmd];

    let submit_info = SubmitInfo::builder()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers_to_submit)
        .signal_semaphores(&signal_semaphores);

    device.queue_submit(
        graphics_queue,
        &[*submit_info],
        sync.in_flight_fence,  // signal this fence when done
    )
    .expect("Failed to submit");

    // --- 6. Present ---
    let swapchains = [swapchain];
    let image_indices = [image_index];
    let present_info = PresentInfoKHR::builder()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    device.queue_present_khr(graphics_queue, &present_info)
        .expect("Failed to present");
    }

    // --- 7. Advance frame index ---
    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}
```

## Step 5: Clean shutdown

Before destroying anything, wait for all frames to finish.

```rust,ignore
unsafe {
    device.device_wait_idle()
        .expect("Failed to wait for device idle");

    for sync in &frame_sync {
        device.destroy_fence(sync.in_flight_fence, None);
        device.destroy_semaphore(sync.image_available, None);
        device.destroy_semaphore(sync.render_finished, None);
    }
}
```

## The synchronization flow

Each frame follows this dependency chain:

```text
wait_for_fences(in_flight_fence)     CPU blocks until frame N-2 is done
        │
acquire_next_image(image_available)  GPU signals when image is ready
        │
reset_fences(in_flight_fence)        Safe to reset now
        │
record commands                      CPU work, no GPU dependency
        │
queue_submit(                        GPU work begins
    wait: image_available,           Wait for image before color output
    signal: render_finished,         Signal when rendering is done
    fence: in_flight_fence           Signal fence when fully complete
)
        │
queue_present(                       Present to screen
    wait: render_finished            Wait for rendering before presenting
)
```

## Common mistakes

**Fence reset before acquire.** If you reset the fence before
`acquire_next_image`, and the acquire call returns an error (e.g.
`OUT_OF_DATE_KHR`), the fence stays unsignaled. The next iteration will
wait on it forever. Always reset the fence *after* a successful acquire.

**Sharing command buffers.** If two frames in flight use the same
command buffer, the CPU might overwrite it while the GPU is still
reading it. Always use one command buffer per frame in flight.

**Forgetting SIGNALED on initial fences.** The loop starts with
`wait_for_fences`. If the fence starts unsignaled, the first frame
deadlocks.

## Notes

- **Triple buffering.** Setting `MAX_FRAMES_IN_FLIGHT = 3` adds one
  more frame of latency but can help if the CPU or GPU has variable
  frame times. Measure before committing to it.
- **Swapchain images vs frames in flight.** The number of swapchain
  images (typically 2 or 3) is independent of `MAX_FRAMES_IN_FLIGHT`.
  Frames in flight control CPU/GPU overlap; swapchain image count
  controls how many images the presentation engine juggles.
- **Resize handling.** When the swapchain is recreated after a window
  resize, you need to wait for all in-flight frames to finish first.
  See [Handle Window Resize](resize.md).
