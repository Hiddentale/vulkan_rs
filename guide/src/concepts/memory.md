# Memory Management

> **Threshold concept.** Vulkan memory management permanently changes how
> you think about GPU resources. In OpenGL, the driver decided where your
> data lived. In Vulkan, *you* decide, and that decision affects
> performance more than almost anything else.

## Motivation

A GPU has multiple memory pools with different properties: some are fast
for the GPU but invisible to the CPU, some are accessible to both but
slower, some are special-purpose. OpenGL hid this complexity behind
`glBufferData` and hoped the driver would make good choices. Sometimes
it did. Often it didn't.

Vulkan exposes this hardware reality directly because the "right" memory
choice depends on your workload, and only you know your workload. A mesh
that never changes after upload needs different memory than a uniform
buffer you update every frame.

## Intuition

### The warehouse analogy

Think of GPU memory like a warehouse with different storage areas:

- **Device-local memory** is the high-speed shelving right next to the
  assembly line (GPU cores). Fast to access, but the front office (CPU)
  can't reach it directly.
- **Host-visible memory** is the loading dock, both the warehouse
  workers (GPU) and the delivery trucks (CPU) can access it, but it's
  slower for the assembly line.
- **Host-coherent memory** is a special loading dock where changes are
  immediately visible to both sides, without needing to shout "new
  stuff here!" (flush/invalidate).
- **Host-cached memory** is a loading dock with a clipboard: the CPU
  reads are fast because they come from a cache, but you need to
  invalidate before reading to make sure the clipboard is up to date.

### The two-step binding model

In Vulkan, creating a Buffer and allocating memory for it are *separate
operations*. This is different from most APIs and is often the first
surprise:

```text
1. Create a Buffer         (describes shape and usage, no memory yet)
2. Query memory requirements (driver tells you: size, alignment, compatible types)
3. Allocate DeviceMemory   (reserve a block from a memory pool)
4. Bind memory to buffer   (connect the two)
```

This separation exists because multiple buffers can share a single
memory allocation (sub-allocation), which is far more efficient than
allocating individually. Production Vulkan applications almost always
use a memory allocator (like VMA) to manage sub-allocation, but
understanding the raw API is essential before using one.

> *Before reading on: why do you think Vulkan separates "create buffer"
> from "allocate memory"? What advantage does this give you that a
> single `create_buffer_with_memory()` call would not?*

### Memory types and heaps

Every Vulkan device exposes a set of **memory heaps** (physical pools of
VRAM or system RAM) and **memory types** (combinations of properties that
describe how a heap can be used).

```text
┌─────────────────────────────────────────────────────────┐
│ Physical Device Memory Properties                       │
│                                                         │
│  Heaps:                                                 │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │ Heap 0: 8 GiB        │  │ Heap 1: 16 GiB           │ │
│  │ flags: DEVICE_LOCAL  │  │ flags: (none)            │ │
│  │ (dedicated GPU VRAM) │  │ (system RAM)             │ │
│  └──────────────────────┘  └──────────────────────────┘ │
│                                                         │
│  Memory Types (each points to a heap):                  │
│  ┌─────────────────────────────────────────────┐        │
│  │ Type 0: heap 0, DEVICE_LOCAL                │        │
│  │ Type 1: heap 1, HOST_VISIBLE | HOST_COHERENT│        │
│  │ Type 2: heap 0, DEVICE_LOCAL | HOST_VISIBLE │  ←BAR  │
│  │ Type 3: heap 1, HOST_VISIBLE | HOST_CACHED  │        │
│  └─────────────────────────────────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The number and properties of heaps and types vary between GPUs. A
discrete GPU typically has separate heaps for VRAM and system RAM. An
integrated GPU often has a single heap that is both device-local and
host-visible. Your code must query these at runtime and choose
accordingly.

### The decision tree

When allocating memory for a resource, follow this logic:

```text
Is this data written by the CPU every frame?
├── Yes → HOST_VISIBLE | HOST_COHERENT
│         (uniform buffers, dynamic vertex data)
│
└── No → Is this data uploaded once and never touched again?
         ├── Yes → DEVICE_LOCAL (use a staging buffer to upload)
         │         (static meshes, textures)
         │
         └── No → Is this data read back by the CPU?
                  ├── Yes → HOST_VISIBLE | HOST_CACHED
                  │         (readback buffers, screenshots)
                  │
                  └── No → DEVICE_LOCAL
                           (render targets, compute output)
```

## Worked example: uploading a mesh to the GPU

This is the most common memory operation in Vulkan: getting vertex data
from the CPU into fast GPU memory. It uses the **staging buffer pattern**.

### Step 1: Create the destination buffer

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

// The buffer that will hold the mesh on the GPU.
// TRANSFER_DST means "this buffer can receive data from a copy command."
let buffer_info = BufferCreateInfo::builder()
    .size(vertex_data_size)
    .usage(
        BufferUsageFlags::VERTEX_BUFFER
        | BufferUsageFlags::TRANSFER_DST
    )
    .sharing_mode(SharingMode::EXCLUSIVE);

let gpu_buffer = unsafe { device.create_buffer(&buffer_info, None)? };
```

### Step 2: Query what memory this buffer needs

```rust,ignore
// The driver tells us: how many bytes, what alignment, and which
// memory types are compatible with this buffer.
let mem_requirements = unsafe {
    device.get_buffer_memory_requirements(gpu_buffer)
};

// mem_requirements.size            → minimum allocation size
// mem_requirements.alignment       → byte alignment requirement
// mem_requirements.memory_type_bits → bitmask of compatible memory types
```

### Step 3: Find the right memory type

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;

// Query what memory the hardware offers.
let mem_properties = unsafe {
    instance.get_physical_device_memory_properties(physical_device)
};

// Find a memory type that is:
//   1. Compatible with the buffer (listed in memory_type_bits)
//   2. Device-local (fast GPU access)
let desired = MemoryPropertyFlags::DEVICE_LOCAL;

let memory_type_index = (0..mem_properties.memory_type_count)
    .find(|&i| {
        let type_compatible =
            mem_requirements.memory_type_bits & (1 << i) != 0;
        let properties_match =
            mem_properties.memory_types[i as usize]
                .property_flags & desired == desired;
        type_compatible && properties_match
    })
    .expect("No suitable memory type found");
```

> *Before reading on: the code above iterates memory types in order
> (0, 1, 2, ...). The Vulkan spec recommends that drivers list memory
> types from most preferred to least preferred. Why does picking the
> first match give you the best performance?*

### Step 4: Allocate and bind

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;

let alloc_info = MemoryAllocateInfo::builder()
    .allocation_size(mem_requirements.size)
    .memory_type_index(memory_type_index);

let gpu_memory = unsafe { device.allocate_memory(&alloc_info, None)? };

// Bind the memory to the buffer. After this, the buffer is backed
// by real memory and can be used.
unsafe { device.bind_buffer_memory(gpu_buffer, gpu_memory, 0)? };
```

### Step 5: Upload via staging buffer

Device-local memory is usually not host-visible, so you can't write to
it directly from the CPU. The solution: create a temporary *staging buffer*
in host-visible memory, write your data there, then copy to the GPU buffer.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

// Create a temporary staging buffer in host-visible memory.
let staging_info = BufferCreateInfo::builder()
    .size(vertex_data_size)
    .usage(BufferUsageFlags::TRANSFER_SRC)
    .sharing_mode(SharingMode::EXCLUSIVE);

let staging_buffer = unsafe { device.create_buffer(&staging_info, None)? };
let staging_reqs = unsafe {
    device.get_buffer_memory_requirements(staging_buffer)
};

// Find HOST_VISIBLE | HOST_COHERENT memory for the staging buffer.
let staging_desired =
    MemoryPropertyFlags::HOST_VISIBLE
    | MemoryPropertyFlags::HOST_COHERENT;

let staging_type_index = (0..mem_properties.memory_type_count)
    .find(|&i| {
        let type_ok = staging_reqs.memory_type_bits & (1 << i) != 0;
        let props_ok =
            mem_properties.memory_types[i as usize]
                .property_flags & staging_desired == staging_desired;
        type_ok && props_ok
    })
    .expect("No host-visible memory type found");

let staging_alloc = MemoryAllocateInfo::builder()
    .allocation_size(staging_reqs.size)
    .memory_type_index(staging_type_index);

let staging_memory = unsafe {
    device.allocate_memory(&staging_alloc, None)?
};
unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? };

// Map the staging memory, copy vertex data in, then unmap.
let mut data_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
unsafe {
    device.map_memory(
        staging_memory,
        0,
        vertex_data_size,
        MemoryMapFlags::empty(),
        &mut data_ptr,
    )?;

    core::ptr::copy_nonoverlapping(
        vertices.as_ptr() as *const u8,
        data_ptr as *mut u8,
        vertex_data_size as usize,
    );

    // Because we chose HOST_COHERENT, we do not need to call
    // flush_mapped_memory_ranges. The write is automatically
    // visible to the GPU.
    device.unmap_memory(staging_memory);
};

// Record a command to copy from staging → gpu buffer.
// (Command buffer recording is covered in the Command Buffers chapter.)
// ... cmd_copy_buffer(staging_buffer, gpu_buffer, &[region]) ...

// After the copy completes on the GPU, clean up the staging buffer.
unsafe {
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);
};
```

### Why not skip the staging buffer?

On some GPUs (especially integrated GPUs and GPUs with Resizable BAR),
there is a memory type that is both `DEVICE_LOCAL` and `HOST_VISIBLE`.
In that case, you *can* map device-local memory directly and skip
the staging buffer. But this memory is often limited in size and not
available on all hardware. The staging buffer pattern works everywhere.

## Formal reference

### Memory property flags

| Flag | Meaning |
|------|---------|
| `DEVICE_LOCAL` | Fastest for GPU access. Usually not host-visible on discrete GPUs. |
| `HOST_VISIBLE` | Can be mapped with `map_memory` for CPU read/write. |
| `HOST_COHERENT` | Mapped writes are automatically visible to the GPU (no flush needed). |
| `HOST_CACHED` | Mapped reads come from CPU cache (fast reads). Requires invalidate before reading GPU-written data. |
| `LAZILY_ALLOCATED` | Memory may not be allocated until used. For transient attachments only. |
| `PROTECTED` | For DRM-protected content. |

### The memory type selection algorithm

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;

fn find_memory_type(
    mem_properties: &PhysicalDeviceMemoryProperties,
    type_bits: u32,       // from MemoryRequirements.memory_type_bits
    desired: MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_properties.memory_type_count).find(|&i| {
        let compatible = type_bits & (1 << i) != 0;
        let has_properties =
            mem_properties.memory_types[i as usize].property_flags
            & desired == desired;
        compatible && has_properties
    })
}
```

This function appears in nearly every Vulkan application. It finds the
first memory type that is compatible with the resource and has the
properties you need.

### Flush and invalidate

If you use memory that is `HOST_VISIBLE` but *not* `HOST_COHERENT`:

- **After writing** from the CPU, call `flush_mapped_memory_ranges` to
  make your writes visible to the GPU.
- **Before reading** on the CPU (after the GPU has written), call
  `invalidate_mapped_memory_ranges` to refresh the CPU's view.

With `HOST_COHERENT` memory, neither call is needed. Most applications
use coherent memory for simplicity.

### Key structs

| Struct | Purpose |
|--------|---------|
| `PhysicalDeviceMemoryProperties` | Describes all heaps and types on the hardware |
| `MemoryType` | One entry: property flags + which heap it draws from |
| `MemoryHeap` | One pool: total size in bytes + heap flags |
| `MemoryRequirements` | What a buffer/image needs: size, alignment, compatible types |
| `MemoryAllocateInfo` | Input to `allocate_memory`: how many bytes, which type |
| `MappedMemoryRange` | Range for flush/invalidate when not using coherent memory |

### Destruction order

```text
1. Ensure GPU is not using the buffer/image (fence or device_wait_idle)
2. Destroy the buffer/image    (device.destroy_buffer / device.destroy_image)
3. Free the memory             (device.free_memory)
```

You must unbind (destroy) all buffers and images from a `DeviceMemory`
before freeing it.

### API reference links

- [`MemoryPropertyFlags`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.MemoryPropertyFlags.html)
- [`PhysicalDeviceMemoryProperties`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.PhysicalDeviceMemoryProperties.html)
- [`MemoryRequirements`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.MemoryRequirements.html)
- [Vulkan spec: Memory Allocation](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#memory)

## Key takeaways

- Vulkan separates buffer/image creation from memory allocation. You
  create the resource, ask what memory it needs, allocate, then bind.
- Memory types have different properties (device-local, host-visible,
  coherent, cached). Choose based on your access pattern.
- The staging buffer pattern (host-visible temp → device-local permanent)
  is the standard way to upload data on discrete GPUs.
- Query memory properties at runtime. Never assume a specific memory
  layout; it varies between GPUs.
- In production, use a sub-allocator (like VMA). Allocating per-buffer
  is correct but slow.
