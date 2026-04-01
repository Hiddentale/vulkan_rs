# Synchronization

> **Threshold concept.** Synchronization is the single most confusing
> aspect of Vulkan for newcomers. Once you understand it, you understand
> Vulkan's execution model. If this chapter takes you three reads, that
> is completely normal.

## Motivation

The GPU does not execute your commands in the order you recorded them.
Not between queues, not between submissions, and not even between draw
calls within the same command buffer. The GPU *pipelines* work: while
one draw call is running its fragment shader, the next draw call might
already be running its vertex shader.

Vulkan gives you **zero implicit ordering guarantees**.

This sounds terrifying, and it is also why Vulkan is fast. The GPU can
overlap operations, reorder for efficiency, and keep all its hardware
units busy. But it means *you* must tell the driver when ordering
matters, because only you know which operations depend on each other.

## Intuition: the factory

Imagine a factory with multiple assembly lines (queue families). Each
line has workers at different stations (pipeline stages) who process
items one after another.

Without synchronization, the factory runs at full speed: items flow
through stations as fast as possible, and different lines operate
independently. This is great, until you have a dependency: "station B
needs the output from station A before it can start."

Vulkan gives you four tools to express these dependencies:

```text
┌─────────┬────────────────────────────┬──────────────────────────┐
│ Tool    │ What it synchronizes       │ Analogy                  │
├─────────┼────────────────────────────┼──────────────────────────┤
│ Fence   │ GPU → CPU                  │ A sign on the factory    │
│         │ "is the GPU done yet?"     │ door: "batch complete"   │
├─────────┼────────────────────────────┼──────────────────────────┤
│ Sema-   │ Queue → Queue              │ A conveyor belt between  │
│ phore   │ "queue B waits for queue A"│ two assembly lines       │
├─────────┼────────────────────────────┼──────────────────────────┤
│ Barrier │ Command → Command          │ A supervisor on one      │
│         │ (within a command buffer)  │ line: "wait for station  │
│         │                            │ A before station B"      │
├─────────┼────────────────────────────┼──────────────────────────┤
│ Event   │ Split barrier              │ A sticky note: "I'll     │
│         │ (signal now, wait later)   │ leave this here, check   │
│         │                            │ for it later"            │
└─────────┴────────────────────────────┴──────────────────────────┘
```

Each tool solves a different problem. Using the wrong tool for the
job is a common source of bugs.

> *Before reading on: you submit two command buffers to the same queue,
> one after the other. Does the second one wait for the first to finish
> before it starts executing?*
>
> Answer: **No.** Submissions to the same queue begin in order, but
> their execution can overlap. The second submission might start its
> vertex shader while the first is still running its fragment shader.
> If you need the first to fully complete before the second starts,
> you need explicit synchronization.

## Worked example 1: CPU waits for GPU (Fence)

**Problem:** You submitted a command buffer. You need to know when
the GPU is done so you can read back the results, or so you can
safely re-record the command buffer for the next frame.

**Solution:** A fence. You pass it to `queue_submit`, and the GPU
signals it when all commands in that submission finish.

```rust,ignore
// ── Create a fence ──────────────────────────────────────────────
//
// SIGNALED means the fence starts in the signaled state.
// This matters for the first frame: wait_for_fences on an
// unsignaled fence with no prior submission would block forever.
let fence_info = vk::FenceCreateInfo::builder()
    .flags(vk::FenceCreateFlags::SIGNALED);

let fence = unsafe { device.create_fence(&fence_info, None)? };
```

```rust,ignore
// ── The render loop ─────────────────────────────────────────────

// Step 1: Wait for the previous frame's GPU work to finish.
//   timeout = u64::MAX means "wait forever"
//   wait_all = 1 (true) means "wait for ALL fences in the slice"
unsafe {
    device.wait_for_fences(&[fence], 1, u64::MAX)?;
};

// Step 2: Reset the fence so it can be signaled again.
//   A fence can only be signaled once. You must reset it
//   before reusing.
unsafe { device.reset_fences(&[fence])? };

// Step 3: Record and submit a command buffer.
// ...record commands...
let submit = vk::SubmitInfo::builder()
    .command_buffers(&[command_buffer]);

// Pass the fence to queue_submit. The GPU will signal it
// when this submission completes.
unsafe {
    device.queue_submit(queue, &[*submit], fence)?;
};

// The CPU continues immediately. The GPU works in parallel.
// Next iteration, wait_for_fences will block until the GPU
// signals this fence.
```

The fence lifecycle:

```text
    create (SIGNALED)
         │
         v
    ┌─> wait ──> reset ──> submit (with fence) ──> GPU signals ─┐
    │                                                           │
    └───────────────────────────────────────────────────────────┘
```

### When to use fences

- Waiting for a frame to finish before re-recording its command buffer
- Waiting for a transfer to complete before reading the result on the CPU
- Throttling the CPU so it doesn't race too far ahead of the GPU

## Worked example 2: Queue-to-queue sync (Semaphore)

**Problem:** The swapchain gives you an image to render into, but the
image might not be ready yet (the display might still be reading it).
After rendering, you need to present the image, but only after the
render commands finish.

**Solution:** Two semaphores. One says "the image is ready to render
into." The other says "rendering is done, safe to present."

```rust,ignore
// Create two semaphores (no flags needed).
let sem_info = vk::SemaphoreCreateInfo::builder();
let image_available = unsafe { device.create_semaphore(&sem_info, None)? };
let render_finished = unsafe { device.create_semaphore(&sem_info, None)? };
```

```rust,ignore
// ── Acquire a swapchain image ───────────────────────────────────
//
// This signals image_available when the image is ready.
let image_index = unsafe {
    device.acquire_next_image_khr(
        swapchain,
        u64::MAX,          // timeout
        image_available,   // semaphore to signal
        vk::Fence::null(), // no fence needed here
    )?
};

// ── Submit rendering commands ───────────────────────────────────
//
// Wait on image_available (at the COLOR_ATTACHMENT_OUTPUT stage,
// because that's when we actually write to the image).
// Signal render_finished when done.
let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

let submit = vk::SubmitInfo::builder()
    .wait_semaphores(&[image_available])
    .wait_dst_stage_mask(&wait_stages)
    .command_buffers(&[command_buffer])
    .signal_semaphores(&[render_finished]);

unsafe {
    device.queue_submit(queue, &[*submit], frame_fence)?;
};

// ── Present the image ───────────────────────────────────────────
//
// Wait on render_finished before the display reads the image.
let present_info = vk::PresentInfoKHR::builder()
    .wait_semaphores(&[render_finished])
    .swapchains(&[swapchain])
    .image_indices(&[image_index]);

unsafe { device.queue_present_khr(queue, &present_info)? };
```

The semaphore flow:

```text
acquire_next_image ──signals──> image_available
                                      │
                                      │ (GPU waits at COLOR_ATTACHMENT_OUTPUT)
                                      v
                               queue_submit ──signals──> render_finished
                                                               │
                                                               │ (presentation waits)
                                                               v
                                                        queue_present
```

### Semaphores vs fences

| | Fence | Semaphore |
|---|---|---|
| **Who waits?** | The CPU (via `wait_for_fences`) | The GPU (another queue operation) |
| **Who signals?** | The GPU (via `queue_submit`) | The GPU (via `queue_submit`) |
| **Use case** | CPU needs to know when GPU is done | One GPU operation depends on another |
| **Can you query it?** | Yes (`get_fence_status`) | No (GPU-only) |

> *Before reading on: why does the submit wait at
> `COLOR_ATTACHMENT_OUTPUT` specifically, instead of waiting at
> `TOP_OF_PIPE` (the very beginning)? What work can the GPU do
> before it needs the swapchain image?*
>
> Answer: The vertex shader, tessellation, and geometry stages do not
> write to the swapchain image. They can run while the image is still
> being read by the display. Only the color attachment output stage
> needs the image, so we delay waiting until that point. This lets the
> GPU overlap more work.

## Worked example 3: Image layout transition (Pipeline Barrier)

**Problem:** You want to copy data into an image, then sample it in
a fragment shader. The image must be in `TRANSFER_DST_OPTIMAL` layout
for the copy, then transitioned to `SHADER_READ_ONLY_OPTIMAL` for
sampling. The GPU must finish the copy before the transition, and
finish the transition before the shader reads.

**Solution:** A pipeline barrier with an image memory barrier.

```rust,ignore
// Transition image from TRANSFER_DST to SHADER_READ_ONLY.
//
// This barrier says:
//   "All TRANSFER_WRITE operations in the TRANSFER stage must
//    complete before any SHADER_READ operations in the
//    FRAGMENT_SHADER stage can begin. Also, change the image
//    layout."
let barrier = vk::ImageMemoryBarrier::builder()
    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
    .dst_access_mask(vk::AccessFlags::SHADER_READ)
    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
    .image(texture_image)
    .subresource_range(vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    });

unsafe {
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,          // src stage
        vk::PipelineStageFlags::FRAGMENT_SHADER,    // dst stage
        vk::DependencyFlags::empty(),
        &[],           // no memory barriers
        &[],           // no buffer memory barriers
        &[*barrier],   // one image memory barrier
    );
};
```

### Understanding the three parts of a barrier

A pipeline barrier has three components that work together:

```text
 1. Stage mask:    WHEN must things happen?
                   "Transfer stage must finish before fragment shader starts"

 2. Access mask:   WHAT memory operations are involved?
                   "Writes from transfers must be visible to shader reads"

 3. Layout:        HOW should the image be reorganized?
                   "Convert from transfer-optimal to shader-read-optimal tiling"
```

All three are needed. The stage mask creates an *execution dependency*
(ordering of operations). The access mask creates a *memory dependency*
(visibility of writes). The layout transition physically reorganizes
how the image data is stored in memory.

A common mistake is setting the stage masks correctly but forgetting
the access masks, or vice versa. Both are required for correctness.

### Common barrier recipes

| From | To | src stage | dst stage | src access | dst access |
|------|----|-----------|-----------|------------|------------|
| Transfer → Shader read | `TRANSFER_DST` → `SHADER_READ_ONLY` | `TRANSFER` | `FRAGMENT_SHADER` | `TRANSFER_WRITE` | `SHADER_READ` |
| Undefined → Transfer dst | `UNDEFINED` → `TRANSFER_DST` | `TOP_OF_PIPE` | `TRANSFER` | `NONE` | `TRANSFER_WRITE` |
| Undefined → Color attachment | `UNDEFINED` → `COLOR_ATTACHMENT` | `TOP_OF_PIPE` | `COLOR_ATTACHMENT_OUTPUT` | `NONE` | `COLOR_ATTACHMENT_WRITE` |
| Color attachment → Present | `COLOR_ATTACHMENT` → `PRESENT_SRC` | `COLOR_ATTACHMENT_OUTPUT` | `BOTTOM_OF_PIPE` | `COLOR_ATTACHMENT_WRITE` | `NONE` |
| Color attachment → Shader read | `COLOR_ATTACHMENT` → `SHADER_READ_ONLY` | `COLOR_ATTACHMENT_OUTPUT` | `FRAGMENT_SHADER` | `COLOR_ATTACHMENT_WRITE` | `SHADER_READ` |

Keep this table handy. Most applications only need these transitions.

## Pipeline stages: the execution order

To understand stage masks, you need to know the order the GPU processes
work. Here is the graphics pipeline, simplified:

```text
TOP_OF_PIPE                    (pseudo-stage: "before anything")
    │
    v
DRAW_INDIRECT                  (read indirect draw parameters)
    │
    v
VERTEX_INPUT                   (read vertex/index buffers)
    │
    v
VERTEX_SHADER                  (run vertex shader)
    │
    v
EARLY_FRAGMENT_TESTS           (depth/stencil test before fragment shader)
    │
    v
FRAGMENT_SHADER                (run fragment shader)
    │
    v
LATE_FRAGMENT_TESTS            (depth/stencil test after fragment shader)
    │
    v
COLOR_ATTACHMENT_OUTPUT        (write to color attachments)
    │
    v
BOTTOM_OF_PIPE                 (pseudo-stage: "after everything")


Special stages (not in the pipeline order):
  TRANSFER                     (copy/blit/clear operations)
  COMPUTE_SHADER               (compute dispatch)
  HOST                         (CPU reads/writes to mapped memory)
  ALL_GRAPHICS                 (shorthand for all graphics stages)
  ALL_COMMANDS                 (shorthand for everything)
```

When you set `src_stage = TRANSFER` and `dst_stage = FRAGMENT_SHADER`,
you are saying: "everything in the TRANSFER stage that came before
this barrier must finish before anything in the FRAGMENT_SHADER stage
that comes after this barrier can start."

## Events: split barriers

Events are an advanced optimization. A pipeline barrier creates a
dependency at a single point in the command buffer. An event lets you
*split* the barrier: signal it at one point, wait for it at a later
point. This gives the GPU more room to reorder work between the signal
and the wait.

```rust,ignore
// Signal the event after the transfer completes.
unsafe {
    device.cmd_set_event(
        command_buffer,
        event,
        vk::PipelineStageFlags::TRANSFER,
    );
};

// ... other commands that don't depend on the transfer ...

// Wait for the event before the fragment shader reads.
unsafe {
    device.cmd_wait_events(
        command_buffer,
        &[event],
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        &[], &[], &[*image_barrier],
    );
};
```

Most applications do not need events. Use pipeline barriers until
profiling shows you need the extra overlap.

## Formal reference

### Synchronization primitives

| Primitive | Scope | Signal | Wait | Create | Destroy |
|-----------|-------|--------|------|--------|---------|
| Fence | GPU → CPU | `queue_submit` | `wait_for_fences` | `create_fence` | `destroy_fence` |
| Semaphore | Queue → Queue | `queue_submit` (signal) | `queue_submit` (wait) | `create_semaphore` | `destroy_semaphore` |
| Pipeline Barrier | Within command buffer | N/A | N/A | N/A (inline command) | N/A |
| Event | Split barrier | `cmd_set_event` | `cmd_wait_events` | `create_event` | `destroy_event` |

### FenceCreateFlags

| Flag | Meaning |
|------|---------|
| `SIGNALED` | Fence starts in the signaled state. Use this for the first frame so `wait_for_fences` doesn't block forever. |

### PipelineStageFlags (most used)

| Flag | When it runs |
|------|-------------|
| `TOP_OF_PIPE` | Before any work. Used as src when there's nothing to wait for. |
| `VERTEX_INPUT` | Reading vertex/index buffers. |
| `VERTEX_SHADER` | Running the vertex shader. |
| `FRAGMENT_SHADER` | Running the fragment shader. |
| `COLOR_ATTACHMENT_OUTPUT` | Writing to color attachments. |
| `TRANSFER` | Copy, blit, and clear operations. |
| `COMPUTE_SHADER` | Running compute shaders. |
| `BOTTOM_OF_PIPE` | After all work. Used as dst when nothing needs to wait. |
| `ALL_COMMANDS` | Shorthand for every stage. Correct but may be slower than a precise mask. |

### AccessFlags (most used)

| Flag | What it protects |
|------|-----------------|
| `VERTEX_ATTRIBUTE_READ` | Vertex shader reads from vertex buffers. |
| `UNIFORM_READ` | Shader reads from uniform buffers. |
| `SHADER_READ` | Shader reads (sampled images, storage buffers). |
| `SHADER_WRITE` | Shader writes (storage images, storage buffers). |
| `COLOR_ATTACHMENT_READ` | Reading color attachments (e.g., blending). |
| `COLOR_ATTACHMENT_WRITE` | Writing to color attachments. |
| `TRANSFER_READ` | Source of a copy/blit. |
| `TRANSFER_WRITE` | Destination of a copy/blit. |
| `HOST_READ` | CPU reads from mapped memory. |
| `HOST_WRITE` | CPU writes to mapped memory. |

### ImageLayout values

| Layout | Optimized for |
|--------|--------------|
| `UNDEFINED` | Nothing. Contents are discarded. Use as old_layout when you don't care about existing data. |
| `GENERAL` | Anything, but not optimal for anything. Last resort. |
| `COLOR_ATTACHMENT_OPTIMAL` | Writing as a color attachment (rendering). |
| `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` | Writing as a depth/stencil attachment. |
| `SHADER_READ_ONLY_OPTIMAL` | Sampling in a shader. |
| `TRANSFER_SRC_OPTIMAL` | Source of a copy/blit. |
| `TRANSFER_DST_OPTIMAL` | Destination of a copy/blit. |
| `PRESENT_SRC` | Presentation to the display (swapchain). |

### The happens-before relationship

Vulkan defines ordering through *execution dependencies* and *memory
dependencies*:

- An **execution dependency** guarantees that operations in the
  *first synchronization scope* (src) complete before operations in
  the *second synchronization scope* (dst) begin.
- A **memory dependency** guarantees that writes in the first access
  scope are *visible* to reads in the second access scope.

Both are needed. Without the execution dependency, operations might
overlap. Without the memory dependency, caches might serve stale data
even after the operation completes.

### API reference links

- [`Fence`](https://docs.rs/vk-engine/latest/vk_engine/vk/struct.Fence.html)
- [`Semaphore`](https://docs.rs/vk-engine/latest/vk_engine/vk/struct.Semaphore.html)
- [`PipelineStageFlags`](https://docs.rs/vk-engine/latest/vk_engine/vk/struct.PipelineStageFlags.html)
- [`AccessFlags`](https://docs.rs/vk-engine/latest/vk_engine/vk/struct.AccessFlags.html)
- [`ImageMemoryBarrier`](https://docs.rs/vk-engine/latest/vk_engine/vk/struct.ImageMemoryBarrier.html)
- [Vulkan spec: Synchronization](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#synchronization)

## Common mistakes

1. **Forgetting access masks.** Stage masks alone create execution
   dependencies, but GPU caches can still serve stale data. You need
   access masks for memory visibility.

2. **Using `ALL_COMMANDS` / `ALL_GRAPHICS` everywhere.** Correct, but
   overly broad. The GPU can't overlap anything across a full-pipeline
   barrier. Use precise stages for better performance.

3. **Reusing a fence without resetting it.** A signaled fence stays
   signaled forever. `wait_for_fences` returns immediately on an
   already-signaled fence. Always `reset_fences` before resubmitting.

4. **Submitting while a command buffer is still pending.** If the GPU
   hasn't finished with a command buffer, you cannot re-record it.
   Wait for its fence first.

5. **Missing the initial fence SIGNALED flag.** On the first frame,
   there is no prior submission to signal the fence. Creating with
   `SIGNALED` avoids an infinite wait.

## Key takeaways

- The GPU does not execute commands in order. You must add explicit
  synchronization where ordering matters.
- **Fences** let the CPU wait for the GPU. **Semaphores** let one GPU
  operation wait for another. **Barriers** order commands within a
  command buffer.
- Barriers have three parts: *when* (stage masks), *what* (access masks),
  and *how* (layout transitions). All three are needed.
- Start with the common barrier recipes table. Most applications only
  need a handful of transitions.
- When in doubt, use broader stages (`ALL_COMMANDS`) to get correct
  behavior first, then narrow down for performance later.
