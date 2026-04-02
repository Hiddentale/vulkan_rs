# Map C Vulkan Calls to vulkan_rs

> **Task:** You have C Vulkan code (or you are reading the Vulkan spec)
> and want to find the equivalent `vulkan_rs` API.

This page is a translation reference. It covers the naming rules, the
structural patterns that differ between C and Rust, and a lookup table
for the most common API calls.

## Naming conventions

### Functions

Strip the `vk` prefix, convert to `snake_case`, and call as a method
on the parent object (`Device` or `Instance`):

| C | vulkan_rs |
|---|-----------|
| `vkCreateBuffer(device, ...)` | `device.create_buffer(...)` |
| `vkCmdDraw(commandBuffer, ...)` | `device.cmd_draw(command_buffer, ...)` |
| `vkEnumeratePhysicalDevices(instance, ...)` | `instance.enumerate_physical_devices()` |
| `vkDestroyPipeline(device, ...)` | `device.destroy_pipeline(...)` |

Note that `vkCmd*` functions take the `CommandBuffer` as a parameter
but are still called on `Device`, not on the command buffer handle.

### Types

Strip the `Vk` prefix. Types live in submodules of `vk`:

| C | vulkan_rs |
|---|-----------|
| `VkBuffer` | `vk::handles::Buffer` |
| `VkBufferCreateInfo` | `vk::structs::BufferCreateInfo` |
| `VkPhysicalDeviceProperties` | `vk::structs::PhysicalDeviceProperties` |
| `VkInstance` | `vk::handles::Instance` (the raw handle) |

Use `use vk::structs::*` or `use vk::handles::*` to bring them into
scope without the module prefix.

### Enum variants

Strip the type prefix and keep `SCREAMING_CASE`:

| C | vulkan_rs |
|---|-----------|
| `VK_FORMAT_R8G8B8A8_SRGB` | `vk::enums::Format::R8G8B8A8_SRGB` |
| `VK_IMAGE_LAYOUT_UNDEFINED` | `vk::enums::ImageLayout::UNDEFINED` |
| `VK_PRESENT_MODE_FIFO_KHR` | `vk::enums::PresentModeKHR::FIFO` |
| `VK_SUCCESS` | `vk::enums::Result::SUCCESS` |

### Bitmask flags

Strip the type prefix and the `_BIT` suffix:

| C | vulkan_rs |
|---|-----------|
| `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT` | `vk::bitmasks::BufferUsageFlags::VERTEX_BUFFER` |
| `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT` | `vk::bitmasks::ImageUsageFlags::COLOR_ATTACHMENT` |
| `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT` | `vk::bitmasks::PipelineStageFlags::FRAGMENT_SHADER` |

Combine flags with the `|` operator, just like in C:

```rust,ignore
use vulkan_rs::vk;
use vk::bitmasks::*;

let usage = BufferUsageFlags::VERTEX_BUFFER
    | BufferUsageFlags::TRANSFER_DST;
```

### Extension names

```rust,ignore
// C:    VK_KHR_SWAPCHAIN_EXTENSION_NAME
// Rust: generated constants in vk::extension_names
use vulkan_rs::vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME;
let device_extensions = [KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr()];
```

## Structural patterns

### Struct initialization

C uses designated initializers. `vulkan_rs` uses the builder pattern,
which auto-fills `sType` and zeroes all other fields:

```c
// C
VkBufferCreateInfo info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .pNext = NULL,
    .size = 1024,
    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
};
VkBuffer buffer;
vkCreateBuffer(device, &info, NULL, &buffer);
```

```rust,ignore
// vulkan_rs
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlags::VERTEX_BUFFER)
    .sharing_mode(SharingMode::EXCLUSIVE);
let buffer = unsafe { device.create_buffer(&info, None) }
    .expect("Failed to create buffer");
```

Key differences:
- `sType` is set automatically by `::builder()`.
- `pNext` defaults to null (use `push_next()` to chain extensions).
- The result is returned, not written through an output pointer.
- The allocator callback (`NULL` in C) becomes `None`.

### pNext extension chains

In C, you manually link structs through `pNext`:

```c
// C
VkPhysicalDeviceVulkan12Features features12 = {
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    .pNext = NULL,
    .bufferDeviceAddress = VK_TRUE,
};
VkDeviceCreateInfo info = {
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = &features12,
    // ...
};
```

In `vulkan_rs`, use `push_next()`:

```rust,ignore
// vulkan_rs
use vulkan_rs::vk;
use vk::structs::*;

let mut features12 = *PhysicalDeviceVulkan12Features::builder()
    .buffer_device_address(1);  // VkBool32: 1 = true
let info = DeviceCreateInfo::builder()
    .push_next(&mut features12)
    .queue_create_infos(&queue_infos);
```

`push_next` is type-safe: you can only chain structs the Vulkan spec
allows for that parent struct.

### The two-call enumerate pattern

Many C Vulkan functions require two calls: one to get the count, one
to fill the array:

```c
// C: two calls to enumerate physical devices
uint32_t count = 0;
vkEnumeratePhysicalDevices(instance, &count, NULL);
VkPhysicalDevice* devices = malloc(count * sizeof(VkPhysicalDevice));
vkEnumeratePhysicalDevices(instance, &count, devices);
```

In `vulkan_rs`, these return a `Vec` directly:

```rust,ignore
// vulkan_rs: one call, returns Vec
let devices = unsafe { instance.enumerate_physical_devices() }
    .expect("Failed to enumerate devices");
```

The crate handles the two-call pattern internally.

### Output parameters

C Vulkan uses pointer parameters for output values. `vulkan_rs`
returns them as `VkResult<T>` or plain `T`:

```c
// C
VkBuffer buffer;
VkResult result = vkCreateBuffer(device, &info, NULL, &buffer);
if (result != VK_SUCCESS) { /* handle error */ }
```

```rust,ignore
// vulkan_rs
use vulkan_rs::vk;
use vk::handles::*;

let buffer: Buffer = unsafe { device.create_buffer(&info, None) }
    .expect("Failed to create buffer");
```

Functions that output multiple handles (like `vkAllocateCommandBuffers`)
return a `Vec` directly:

```rust,ignore
use vulkan_rs::vk;
use vk::handles::*;

let cmd_buffers = unsafe {
    device.allocate_command_buffers(&alloc_info)
}
.expect("Failed to allocate command buffers");
```

## Search tip: `#[doc(alias)]`

All `vulkan_rs` types and functions carry `#[doc(alias = "vkOriginalName")]`
attributes. If you know the C name, type it into the rustdoc search bar
and it will find the Rust equivalent. For example, searching for
`VkBufferCreateInfo` will find `vk::BufferCreateInfo`.

## Common API mapping table

| C function | vulkan_rs method | Returns |
|------------|-----------------|---------|
| `vkCreateInstance` | `entry.create_instance(&info, None)` | `VkResult<Instance>` |
| `vkDestroyInstance` | `instance.destroy_instance(None)` | `()` |
| `vkEnumeratePhysicalDevices` | `instance.enumerate_physical_devices()` | `VkResult<Vec<PhysicalDevice>>` |
| `vkGetPhysicalDeviceProperties` | `instance.get_physical_device_properties(phys)` | `PhysicalDeviceProperties` |
| `vkGetPhysicalDeviceQueueFamilyProperties` | `instance.get_physical_device_queue_family_properties(phys)` | `Vec<QueueFamilyProperties>` |
| `vkCreateDevice` | `instance.create_device(phys, &info, None)` | `VkResult<Device>` |
| `vkDestroyDevice` | `device.destroy_device(None)` | `()` |
| `vkGetDeviceQueue` | `device.get_device_queue(family, index)` | `Queue` |
| `vkCreateBuffer` | `device.create_buffer(&info, None)` | `VkResult<Buffer>` |
| `vkDestroyBuffer` | `device.destroy_buffer(buffer, None)` | `()` |
| `vkAllocateMemory` | `device.allocate_memory(&info, None)` | `VkResult<DeviceMemory>` |
| `vkFreeMemory` | `device.free_memory(memory, None)` | `()` |
| `vkBindBufferMemory` | `device.bind_buffer_memory(buffer, memory, offset)` | `VkResult<()>` |
| `vkMapMemory` | `device.map_memory(memory, offset, size, flags)` | `VkResult<*mut c_void>` |
| `vkUnmapMemory` | `device.unmap_memory(memory)` | `()` |
| `vkCreateImage` | `device.create_image(&info, None)` | `VkResult<Image>` |
| `vkDestroyImage` | `device.destroy_image(image, None)` | `()` |
| `vkCreateImageView` | `device.create_image_view(&info, None)` | `VkResult<ImageView>` |
| `vkCreateRenderPass` | `device.create_render_pass(&info, None)` | `VkResult<RenderPass>` |
| `vkCreateGraphicsPipelines` | `device.create_graphics_pipelines(cache, &infos, None)` | `VkResult<Vec<Pipeline>>` |
| `vkCreateCommandPool` | `device.create_command_pool(&info, None)` | `VkResult<CommandPool>` |
| `vkAllocateCommandBuffers` | `device.allocate_command_buffers(&info)` | `VkResult<Vec<CommandBuffer>>` |
| `vkBeginCommandBuffer` | `device.begin_command_buffer(cmd, &info)` | `VkResult<()>` |
| `vkEndCommandBuffer` | `device.end_command_buffer(cmd)` | `VkResult<()>` |
| `vkCmdBeginRenderPass` | `device.cmd_begin_render_pass(cmd, &info, contents)` | `()` |
| `vkCmdEndRenderPass` | `device.cmd_end_render_pass(cmd)` | `()` |
| `vkCmdBindPipeline` | `device.cmd_bind_pipeline(cmd, bind_point, pipeline)` | `()` |
| `vkCmdDraw` | `device.cmd_draw(cmd, vertices, instances, first_v, first_i)` | `()` |
| `vkCmdCopyBuffer` | `device.cmd_copy_buffer(cmd, src, dst, &regions)` | `()` |
| `vkQueueSubmit` | `device.queue_submit(queue, &submits, fence)` | `VkResult<()>` |
| `vkQueuePresentKHR` | `device.queue_present_khr(queue, &info)` | `VkResult<()>` |
| `vkDeviceWaitIdle` | `device.device_wait_idle()` | `VkResult<()>` |
| `vkCreateFence` | `device.create_fence(&info, None)` | `VkResult<Fence>` |
| `vkWaitForFences` | `device.wait_for_fences(&fences, wait_all, timeout)` | `VkResult<()>` |
| `vkResetFences` | `device.reset_fences(&fences)` | `VkResult<()>` |
| `vkCreateSemaphore` | `device.create_semaphore(&info, None)` | `VkResult<Semaphore>` |
| `vkCreateDescriptorSetLayout` | `device.create_descriptor_set_layout(&info, None)` | `VkResult<DescriptorSetLayout>` |
| `vkAllocateDescriptorSets` | `device.allocate_descriptor_sets(&info)` | `VkResult<Vec<DescriptorSet>>` |
| `vkUpdateDescriptorSets` | `device.update_descriptor_sets(&writes, &copies)` | `()` |

## Worked example: full translation

### C version

```c
// Create a vertex buffer, bind memory, copy data
VkBufferCreateInfo buf_info = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = sizeof(vertices),
    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
};
VkBuffer buffer;
vkCreateBuffer(device, &buf_info, NULL, &buffer);

VkMemoryRequirements mem_req;
vkGetBufferMemoryRequirements(device, buffer, &mem_req);

VkMemoryAllocateInfo alloc_info = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .allocationSize = mem_req.size,
    .memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
};
VkDeviceMemory memory;
vkAllocateMemory(device, &alloc_info, NULL, &memory);
vkBindBufferMemory(device, buffer, memory, 0);

void* data;
vkMapMemory(device, memory, 0, buf_info.size, 0, &data);
memcpy(data, vertices, sizeof(vertices));
vkUnmapMemory(device, memory);
```

### vulkan_rs version

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

unsafe {
    let buf_info = BufferCreateInfo::builder()
        .size(std::mem::size_of_val(&vertices) as u64)
        .usage(BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(SharingMode::EXCLUSIVE);
    let buffer = device.create_buffer(&buf_info, None)
        .expect("Failed to create buffer");

    let mem_req = device.get_buffer_memory_requirements(buffer);

    let alloc_info = MemoryAllocateInfo::builder()
        .allocation_size(mem_req.size)
        .memory_type_index(find_memory_type(
            mem_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE
                | MemoryPropertyFlags::HOST_COHERENT,
        ));
    let memory = device.allocate_memory(&alloc_info, None)
        .expect("Failed to allocate memory");
    device.bind_buffer_memory(buffer, memory, 0)
        .expect("Failed to bind buffer memory");

    let data = device.map_memory(
        memory, 0, buf_info.size, MemoryMapFlags::empty(),
    )
    .expect("Failed to map memory");
    std::ptr::copy_nonoverlapping(
        vertices.as_ptr() as *const u8,
        data as *mut u8,
        std::mem::size_of_val(&vertices),
    );
    device.unmap_memory(memory);
}
```

The structure is the same: create, query requirements, allocate, bind,
map, copy, unmap. The differences are syntactic, not conceptual.
