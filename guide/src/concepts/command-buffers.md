# Command Buffers

## Motivation

In OpenGL, calling `glDrawArrays` immediately sends work to the GPU
(or at least, the driver pretends it does). In Vulkan, you *record*
commands into a buffer, then *submit* that buffer to a queue. The GPU
processes the queue asynchronously while your CPU moves on.

This separation exists for three reasons:

1. **Batching.** One submission of many commands is cheaper than many
   individual calls. Each submission has overhead (kernel transitions,
   driver bookkeeping), so bundling hundreds of draw calls into a single
   command buffer and submitting once is dramatically faster.
2. **Reuse.** You can record a command buffer once and submit it many
   times. If a scene doesn't change, why re-record every frame?
3. **Multi-threading.** Different CPU threads can record into different
   command buffers simultaneously, then submit them all on one thread.
   This is how modern engines scale across CPU cores.

## Intuition

### The shopping list analogy

A command buffer is a shopping list. You write down everything you need
("bind this pipeline", "draw 36 vertices", "copy this image"), then
hand the list to someone else (a GPU queue) who goes and does it all.
You don't stand in the store waiting for each item, you hand off the
list and do other work.

The lifecycle looks like this:

```text
┌────────────┐     ┌────────────┐     ┌────────────┐
│   Record   │────>│   Submit   │────>│  Execute   │
│  (CPU)     │     │  (CPU→GPU) │     │  (GPU)     │
│            │     │            │     │            │
│ "bind X"   │     │ hand off   │     │ GPU reads  │
│ "draw Y"   │     │ to queue   │     │ the list   │
│ "copy Z"   │     │            │     │ and acts   │
└────────────┘     └────────────┘     └────────────┘
```

The CPU is free to do other work (including recording the *next*
frame's command buffer) while the GPU executes.

### Command pools: why they exist

Allocating command buffers one at a time would be like allocating
individual bytes from the OS. It's correct, but the overhead per
allocation is huge. Command pools solve this by pre-allocating a
chunk of memory, then handing out command buffers from that pool
cheaply.

```text
┌──────────── Command Pool ────────────┐
│                                      │
│  ┌──────────┐  ┌──────────┐          │
│  │ CmdBuf 0 │  │ CmdBuf 1 │  ...     │
│  └──────────┘  └──────────┘          │
│                                      │
│  (all allocated from one pool)       │
│  (pool is tied to one queue family)  │
└──────────────────────────────────────┘
```

Each pool is tied to a single queue family. This lets the driver
optimize the memory layout for that queue type.

> *Before reading on: if command pools are tied to a single queue
> family, and you want to record commands for both a graphics queue
> and a transfer queue, how many pools do you need?*

### Primary vs secondary command buffers

**Primary** command buffers are what you submit to queues. They can
contain any command.

**Secondary** command buffers cannot be submitted directly. Instead,
they are *executed* from within a primary command buffer using
`cmd_execute_commands`. Think of them as subroutines: you record
reusable chunks of work (like "render the UI") into secondary buffers,
then call them from your primary buffer.

```text
Primary command buffer:
  begin render pass
  bind pipeline A
  draw meshes
  execute_commands(secondary_ui_buffer)   ← calls the secondary
  end render pass
```

Most applications start with primary buffers only and add secondary
buffers when they need multi-threaded recording or reusable sub-passes.

## Worked example: record and submit

This example creates a command pool, allocates a command buffer,
records a simple buffer copy, and submits it.

### Step 1: Create a command pool

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Create a pool for the graphics queue family.
// RESET_COMMAND_BUFFER lets us reset individual command buffers
// instead of resetting the entire pool.
let pool_info = CommandPoolCreateInfo::builder()
    .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
    .queue_family_index(graphics_queue_family);

let command_pool = unsafe {
    device.create_command_pool(&pool_info, None)?
};
```

### Step 2: Allocate a command buffer

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Allocate one primary command buffer from the pool.
let alloc_info = CommandBufferAllocateInfo::builder()
    .command_pool(command_pool)
    .level(CommandBufferLevel::PRIMARY)
    .command_buffer_count(1);

// allocate_command_buffers returns a Vec of handles.
let command_buffer = unsafe {
    device.allocate_command_buffers(&alloc_info)?
}[0];
```

### Step 3: Record commands

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Begin recording. ONE_TIME_SUBMIT tells the driver this buffer
// will be submitted once and then reset or freed, enabling
// driver-side optimizations.
let begin_info = CommandBufferBeginInfo::builder()
    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

unsafe {
    device.begin_command_buffer(command_buffer, &begin_info)?;
};

// Record a buffer copy command.
// This does NOT execute the copy. It records the instruction
// into the command buffer for later execution.
let copy_region = BufferCopy {
    src_offset: 0,
    dst_offset: 0,
    size: 1024,
};

unsafe {
    device.cmd_copy_buffer(
        command_buffer,
        src_buffer,
        dst_buffer,
        &[copy_region],
    );
};

// Finish recording.
unsafe { device.end_command_buffer(command_buffer)? };
```

> *Before reading on: between `begin_command_buffer` and
> `end_command_buffer`, the command buffer is in the "recording"
> state. What do you think happens if you try to submit a command
> buffer that is still in the recording state?*

### Step 4: Submit to a queue

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Build a submit info. This describes:
//   - which command buffers to execute
//   - which semaphores to wait on before starting
//   - which semaphores to signal when done
let submit_info = SubmitInfo::builder()
    .command_buffers(&[command_buffer]);

// Submit to the graphics queue.
// The Fence (here Fence::null()) will be signaled when the GPU
// finishes all commands in this submission. Passing null means
// "I don't need to know when it's done from the CPU."
unsafe {
    device.queue_submit(
        graphics_queue,
        &[*submit_info],
        Fence::null(),
    )?;
};

// For this example, we wait for the queue to finish before
// continuing. In a real application, you would use a fence
// instead of blocking the CPU.
unsafe { device.queue_wait_idle(graphics_queue)? };
```

### Step 5: Clean up

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// Option A: Free the command buffer back to the pool.
unsafe {
    device.free_command_buffers(command_pool, &[command_buffer]);
};

// Option B: Reset for reuse (only if pool was created with
// RESET_COMMAND_BUFFER flag).
unsafe {
    device.reset_command_buffer(
        command_buffer,
        CommandBufferResetFlags::empty(),
    )?;
};

// When you're done with the pool entirely:
unsafe { device.destroy_command_pool(command_pool, None) };
// This implicitly frees all command buffers allocated from it.
```

## Command buffer states

A command buffer is always in one of these states:

```text
                  allocate
   ┌────────────────────────────────┐
   v                                │
 Initial ──begin──> Recording ──end──> Executable ──submit──> Pending
   ^                                      │                      │
   │                                      │                      │
   └──────────── reset ───────────────────┘       (GPU finishes) |
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
                    (returns to Executable or Initial)
```

| State | What you can do |
|-------|-----------------|
| **Initial** | Nothing useful. Call `begin_command_buffer` to start recording. |
| **Recording** | Record commands (`cmd_*` methods). Call `end_command_buffer` when done. |
| **Executable** | Submit to a queue. Or reset to record again. |
| **Pending** | The GPU is executing it. Do not touch it. Wait for completion. |

The most common mistake is trying to re-record or reset a command buffer
while it is still pending (the GPU hasn't finished yet). Validation
layers will catch this.

## Common patterns

### One-shot command buffer for transfers

Many operations (uploading textures, transitioning image layouts) need
a command buffer just once. The pattern:

```rust,ignore
use vulkan_rust::vk;
use vk::*;

unsafe fn one_shot_submit(
    device: &Device,
    pool: CommandPool,
    queue: Queue,
    record: impl FnOnce(CommandBuffer),
) -> VkResult<()> {
    // Allocate
    let alloc_info = CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

    // Record
    let begin = CommandBufferBeginInfo::builder()
        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin)? };
    record(cmd);
    unsafe { device.end_command_buffer(cmd)? };

    // Submit and wait
    let submit = SubmitInfo::builder()
        .command_buffers(&[cmd]);
    unsafe {
        device.queue_submit(queue, &[*submit], Fence::null())?;
        device.queue_wait_idle(queue)?;
    };

    // Free
    unsafe { device.free_command_buffers(pool, &[cmd]) };
    Ok(())
}
```

This is the pattern used for staging buffer uploads in the
[Memory Management](memory.md) chapter.

### Per-frame command buffers

For rendering, you typically have one command buffer per frame in flight:

```text
Frame 0: [record on CPU] ──submit──> [execute on GPU]
Frame 1: [record on CPU] ──submit──> [execute on GPU]
          ↑                              ↑
          recording while               executing the
          GPU runs the                  commands we
          previous frame                just submitted
```

Each frame waits for its fence before re-recording. See
[Synchronization](synchronization.md) for how fences and semaphores
coordinate this.

## Formal reference

### Command pool creation flags

| Flag | Meaning |
|------|---------|
| `TRANSIENT` | Hint: command buffers from this pool are short-lived. Lets the driver optimize allocation. |
| `RESET_COMMAND_BUFFER` | Allows individual command buffers to be reset. Without this, you can only reset the entire pool. |
| `PROTECTED` | Command buffers allocated from this pool can operate on protected resources. |

### Command buffer begin flags

| Flag | Meaning |
|------|---------|
| `ONE_TIME_SUBMIT` | This buffer will be submitted once, then reset or freed. Enables driver optimizations. |
| `RENDER_PASS_CONTINUE` | Secondary command buffer: this will be entirely inside a render pass. |
| `SIMULTANEOUS_USE` | This buffer can be submitted to multiple queues or resubmitted while still pending. |

### Recording methods on Device

All recording methods follow the pattern `device.cmd_*(command_buffer, ...)`.
The `device` dispatches to the correct function pointer, the `command_buffer`
identifies which buffer to record into. Examples:

| Method | Purpose |
|--------|---------|
| `cmd_bind_pipeline(cb, bind_point, pipeline)` | Set the active pipeline |
| `cmd_draw(cb, vertices, instances, first_vert, first_inst)` | Draw without an index buffer |
| `cmd_copy_buffer(cb, src, dst, &[regions])` | Copy between buffers |
| `cmd_begin_render_pass(cb, &begin_info, contents)` | Start a render pass |
| `cmd_end_render_pass(cb)` | End the current render pass |

The full list has ~150 `cmd_*` methods covering every Vulkan command.

### Destruction rules

1. **Wait for the GPU before freeing.** A command buffer in the Pending
   state must not be freed or reset. Use a fence or `device_wait_idle`.
2. **Destroying a pool frees all its buffers.** You do not need to free
   command buffers individually before destroying their pool.
3. **Pools are not thread-safe.** If two threads record command buffers
   from the same pool, you must synchronize externally. The typical
   solution: one pool per thread.

### SubmitInfo structure

`SubmitInfo` connects command buffers to synchronization primitives:

```text
SubmitInfo {
    wait_semaphores    + wait_dst_stage_mask   ← "wait for these before starting"
    command_buffers                             ← "execute these"
    signal_semaphores                           ← "signal these when done"
}
```

The `wait_dst_stage_mask` specifies *which pipeline stages* must wait,
not the entire submission. This enables the GPU to start early stages
while still waiting for a semaphore on a later stage.

### API reference links

- [`CommandPool`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.CommandPool.html)
- [`CommandBuffer`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.CommandBuffer.html)
- [`CommandPoolCreateFlags`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.CommandPoolCreateFlags.html)
- [`SubmitInfo`](https://docs.rs/vulkan-rust/latest/vulkan_rust/vk/struct.SubmitInfo.html)
- [Vulkan spec: Command Buffers](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#commandbuffers)

## Key takeaways

- Commands are recorded, not executed. Recording is cheap CPU work;
  execution happens asynchronously on the GPU.
- Command pools amortize allocation cost. One pool per queue family,
  typically one pool per thread.
- Command buffers have states: Initial → Recording → Executable → Pending.
  Never touch a Pending buffer.
- Use `ONE_TIME_SUBMIT` for throw-away work (uploads, transitions).
  Use per-frame buffers with fences for rendering.
- The `SubmitInfo` struct is where command buffers meet synchronization.
  That connection is the topic of the [next chapter](synchronization.md).
