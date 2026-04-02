# Port from ash to vulkan_rs

> **Task:** Migrate an existing `ash`-based project to `vulkan_rs`
> (published as `vulkan-rs` on crates.io).

If you already have a working `ash` project, switching to `vulkan_rs`
is mostly mechanical. The Vulkan concepts are identical, and the API
surface maps one-to-one. This guide covers every difference you will
encounter.

## What stays the same

Before diving into differences, note what does *not* change:

- All Vulkan functions are `unsafe`.
- You must explicitly destroy every object you create (no RAII/Drop on handles).
- Handles are lightweight `Copy` types.
- The same Vulkan mental model applies: instances, devices, queues,
  command buffers, pipelines, descriptor sets, synchronization primitives.

## Key differences at a glance

| Aspect | ash | vulkan_rs |
|--------|-----|-----------|
| Crate name | `ash` | `vulkan-rs` |
| Command style | Trait methods (`DeviceV1_0`, `KhrSwapchainFn`) | Inherent methods on `Device` / `Instance` |
| Trait imports | One per API version + one per extension | None needed |
| Raw types | `ash::vk::*` | `vulkan_rs::vk::*` |
| Builders | `::builder()` returns `Builder`, call `.build()` | `::builder()` returns `Builder` that derefs to inner struct |
| Extensions | Manual loader structs (`ash::khr::swapchain::Device`) | All loaded automatically, call methods on `Device` directly |
| Interop | Limited `from_raw` on some types | `Instance::from_raw_parts` / `Device::from_raw_parts` |
| Error type | `ash::vk::Result` with separate success/error enums | `VkResult<T>` wrapping `vk::Result` |

## Step 1: Replace the Cargo dependency

```toml
# Before (ash)
[dependencies]
ash = "0.38"

# After (vulkan_rs)
[dependencies]
vulkan-rs = "0.1"
```

## Step 2: Remove trait imports

This is the single biggest ergonomic difference. In `ash`, every
Vulkan API version and extension requires a trait import:

```rust,ignore
// ash: you need these traits in scope to call device methods
use ash::vk;
use ash::Device;
// Without this import, device.create_buffer() does not exist:
use ash::version::DeviceV1_0;
// Without this import, device.create_swapchain_khr() does not exist:
use ash::khr::swapchain::Device as SwapchainDevice;
```

In `vulkan_rs`, every command is an inherent method on `Device` or
`Instance`. No trait imports, no extension loader structs:

```rust,ignore
// vulkan_rs: this is all you need
use vulkan_rs::vk;
use vulkan_rs::Device;
// device.create_buffer() and device.create_swapchain_khr()
// are both available immediately.
```

**Migration action:** Delete all `use ash::version::*` and
`use ash::extensions::*` imports. Replace `use ash::vk` with
`use vulkan_rs::vk`.

## Step 3: Replace Entry, Instance, and Device creation

### Entry and Instance

```rust,ignore
// ── ash ─────────────────────────────────────────────────
let entry = ash::Entry::linked();
let app_info = vk::ApplicationInfo::builder()
    .api_version(vk::make_api_version(0, 1, 3, 0))
    .build();
let create_info = vk::InstanceCreateInfo::builder()
    .application_info(&app_info)
    .build();
let instance = unsafe { entry.create_instance(&create_info, None)? };

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::structs::*;

let loader = vulkan_rs::LibloadingLoader::new()
    .expect("Failed to load Vulkan");
let entry = unsafe { vulkan_rs::Entry::new(loader) }
    .expect("Failed to create entry");

let app_info = ApplicationInfo::builder()
    .api_version((1 << 22) | (3 << 12));  // Vulkan 1.3
let create_info = InstanceCreateInfo::builder()
    .p_application_info(&app_info);
let instance = unsafe { entry.create_instance(&create_info, None) }
    .expect("Failed to create instance");
```

The main changes: `Entry` is loaded through `LibloadingLoader` instead
of `linked()`, `make_api_version` is replaced with a raw `u32`
expression, `.application_info()` becomes `.p_application_info()`, and
`.build()` calls are removed. The builder derefs to the inner struct,
so you can pass `&create_info` directly where a `&InstanceCreateInfo`
is expected.

### Device

```rust,ignore
// ── ash ─────────────────────────────────────────────────
let queue_info = vk::DeviceQueueCreateInfo::builder()
    .queue_family_index(0)
    .queue_priorities(&[1.0])
    .build();
let device_info = vk::DeviceCreateInfo::builder()
    .queue_create_infos(std::slice::from_ref(&queue_info))
    .build();
let device = unsafe {
    instance.create_device(physical_device, &device_info, None)?
};

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::structs::*;

let queue_info = DeviceQueueCreateInfo::builder()
    .queue_family_index(0)
    .queue_priorities(&[1.0]);
let device_info = DeviceCreateInfo::builder()
    .queue_create_infos(std::slice::from_ref(&queue_info));
let device = unsafe {
    instance.create_device(physical_device, &device_info, None)
}
.expect("Failed to create device");
```

## Step 4: Update builders (drop `.build()`)

In `ash`, builders require `.build()` to produce the final struct.
In `vulkan_rs`, builders implement `Deref<Target = T>`, so the
conversion is implicit:

```rust,ignore
// ── ash ─────────────────────────────────────────────────
let info = vk::BufferCreateInfo::builder()
    .size(1024)
    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
    .sharing_mode(vk::SharingMode::EXCLUSIVE)
    .build();  // <-- required in ash

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlags::VERTEX_BUFFER)
    .sharing_mode(SharingMode::EXCLUSIVE);
    // No .build(), pass &info directly to create_buffer()
```

**Migration action:** Search your codebase for `.build()` and remove
every occurrence on Vulkan builder types.

## Step 5: Command buffer recording

The pattern is identical, just without trait imports:

```rust,ignore
// ── ash ─────────────────────────────────────────────────
use ash::version::DeviceV1_0;  // required for begin/end

let begin_info = vk::CommandBufferBeginInfo::builder()
    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
    .build();
unsafe {
    device.begin_command_buffer(cmd, &begin_info)?;
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
    device.cmd_draw(cmd, 3, 1, 0, 0);
    device.end_command_buffer(cmd)?;
}

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

let begin_info = CommandBufferBeginInfo::builder()
    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
unsafe {
    device.begin_command_buffer(cmd, &begin_info)
        .expect("Failed to begin command buffer");
    device.cmd_bind_pipeline(cmd, PipelineBindPoint::GRAPHICS, pipeline);
    device.cmd_draw(cmd, 3, 1, 0, 0);
    device.end_command_buffer(cmd)
        .expect("Failed to end command buffer");
}
```

## Step 6: Queue submission

```rust,ignore
// ── ash ─────────────────────────────────────────────────
let submit_info = vk::SubmitInfo::builder()
    .command_buffers(&[cmd])
    .wait_semaphores(&[image_available])
    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
    .signal_semaphores(&[render_finished])
    .build();
unsafe { device.queue_submit(queue, &[submit_info.build()], fence)? };

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::structs::*;

let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
let cmd_bufs = [cmd];
let wait_sems = [image_available];
let signal_sems = [render_finished];
let submit_info = SubmitInfo::builder()
    .command_buffers(&cmd_bufs)
    .wait_semaphores(&wait_sems)
    .wait_dst_stage_mask(&wait_stages)
    .signal_semaphores(&signal_sems);
unsafe {
    device.queue_submit(queue, &[*submit_info], fence)
        .expect("Failed to submit");
};
```

## Step 7: Error handling

`ash` splits Vulkan results into success codes and error codes.
`vulkan_rs` uses a single `VkResult<T>` type:

```rust,ignore
// ── ash ─────────────────────────────────────────────────
match unsafe { device.create_buffer(&info, None) } {
    Ok(buffer) => { /* ... */ }
    Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => { /* ... */ }
    Err(e) => panic!("Unexpected: {:?}", e),
}

// ── vulkan_rs ───────────────────────────────────────────
use vulkan_rs::vk;
use vk::enums::Result as VkError;

match unsafe { device.create_buffer(&info, None) } {
    Ok(buffer) => { /* ... */ }
    Err(VkError::ERROR_OUT_OF_DEVICE_MEMORY) => { /* ... */ }
    Err(e) => panic!("Unexpected: {e:?}"),
}
```

The match arms look the same. The difference is that `VkResult<T>`
implements `std::error::Error`, so it works with `anyhow`, `eyre`,
and the `?` operator out of the box.

## Step 8: Extensions

In `ash`, extensions require separate loader structs:

```rust,ignore
// ash: manual extension loading
let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
let swapchain = unsafe {
    swapchain_loader.create_swapchain(&create_info, None)?
};
```

In `vulkan_rs`, all extension functions are loaded automatically when
the `Device` or `Instance` is created. You call them as regular methods:

```rust,ignore
// vulkan_rs: no loader, just call the method
let swapchain = unsafe {
    device.create_swapchain_khr(&create_info, None)
}
.expect("Failed to create swapchain");
```

**Migration action:** Delete all extension loader struct construction.
Replace `loader.method()` with `device.method()` or `instance.method()`.

## Step 9: Interop with `from_raw_parts`

If another library (OpenXR, a C plugin, a test harness) gives you raw
Vulkan handles, `vulkan_rs` provides `from_raw_parts` to wrap them:

```rust,ignore
// Wrap an externally-created VkInstance
let instance = unsafe {
    vulkan_rs::Instance::from_raw_parts(raw_instance, get_instance_proc_addr)
};

// Wrap an externally-created VkDevice
let device = unsafe {
    vulkan_rs::Device::from_raw_parts(raw_device, get_device_proc_addr)
};
```

This loads all function pointers from the provided `get_*_proc_addr`,
so the wrapped object works identically to one created through `Entry`.

## Quick-reference migration checklist

- [ ] Replace `ash` with `vulkan-rs` in `Cargo.toml`
- [ ] Replace `use ash::vk` with `use vulkan_rs::vk`
- [ ] Delete all `use ash::version::*` trait imports
- [ ] Delete all extension loader struct construction
- [ ] Remove every `.build()` on Vulkan builder types
- [ ] Replace `ash::Entry` / `ash::Instance` / `ash::Device` with `vulkan_rs::*`
- [ ] Replace extension loader method calls with direct `device.method()` calls
- [ ] Update error handling if you matched on ash-specific error types
- [ ] Compile and fix any remaining type mismatches
