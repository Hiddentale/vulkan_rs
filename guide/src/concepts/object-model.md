# The Vulkan Object Model

## Motivation

Every Vulkan API call operates on *handles*, opaque references to objects
that live on the GPU or in the driver. Before you can do anything useful
in Vulkan, you need to understand what these handles are, how they relate
to each other, and who is responsible for destroying them.

If you have used file descriptors on Unix, database connections, or COM
objects on Windows, the concept is the same: you request a resource, you
get back an opaque identifier, you use that identifier in every subsequent
call, and you close it when you are done. Vulkan has roughly 59 different
handle types, but they all follow this pattern.

## Intuition

### Handles are opaque identifiers, not objects

A Vulkan handle is not a pointer to a struct you can inspect. It is an
opaque number the driver gives you. You pass it back to the driver in
later calls, and the driver uses it to look up the real resource internally.
You never dereference a handle or read its fields.

In `vulkan_rs`, every handle is a `#[repr(transparent)]` newtype over
either `usize` or `u64`:

```rust,ignore
// This is the entire definition of a Buffer handle.
// There is nothing inside it except a number.
#[repr(transparent)]
pub struct Buffer(u64);
```

### Handles form a parent-child tree

Vulkan objects are not independent. They form a hierarchy where each
object is created from (and belongs to) a parent:

```text
Instance                          (your connection to the Vulkan driver)
├── PhysicalDevice                (a GPU on the system, enumerated, not created)
│   └── Device                    (your logical interface to that GPU)
│       ├── Queue                 (a submission endpoint, retrieved, not created)
│       ├── CommandPool
│       │   └── CommandBuffer     (allocated from a pool, not created directly)
│       ├── Buffer
│       ├── Image
│       ├── Fence
│       ├── Semaphore
│       ├── Pipeline
│       ├── DescriptorPool
│       │   └── DescriptorSet    (allocated from a pool, not created directly)
│       └── ... (~50 more types)
└── SurfaceKHR                    (a window's rendering target)
```

This hierarchy determines two things:

1. **Creation order.** You cannot create a `Buffer` without a `Device`,
   and you cannot create a `Device` without a `PhysicalDevice`, which
   requires an `Instance`.
2. **Destruction order.** You must destroy children before their parent.
   If you destroy a `Device` while it still has live `Buffer` handles,
   that is undefined behavior.

> *Before reading on: look at the tree above. Why do you think
> `CommandBuffer` and `DescriptorSet` are "allocated from a pool"
> instead of "created directly" like `Buffer` or `Image`?*

### The creation-destruction lifecycle

Almost every Vulkan object follows the same lifecycle:

```text
1. Fill a CreateInfo struct     (describe what you want)
2. Call create_xxx()            (driver creates it, gives you a handle)
3. Use the handle               (pass it to other API calls)
4. Call destroy_xxx()           (you are done, release it)
```

The exception is objects that are *enumerated* (PhysicalDevice, Queue)
or *allocated from pools* (CommandBuffer, DescriptorSet). These have
slightly different creation/destruction patterns, covered below.

### Dispatchable vs non-dispatchable handles

Vulkan has two categories of handle, and the difference matters for
understanding how the driver works internally.

**Dispatchable handles** (Instance, PhysicalDevice, Device,
CommandBuffer, Queue) are pointer-sized (`usize`). Internally, the
driver stores a dispatch table at the address the handle points to.
When you call a Vulkan function, the loader uses this dispatch table
to route the call to the correct driver. There are only 5 dispatchable
handle types.

**Non-dispatchable handles** (Buffer, Image, Fence, Pipeline, and
all the rest) are 64-bit integers (`u64`). They are opaque identifiers
that the driver interprets however it likes. There are roughly 54 of
these.

You rarely need to think about this distinction in application code.
It matters when you are doing interop (passing handles between
processes or APIs) or when you are debugging driver internals.

## Worked example: the complete lifecycle of a Buffer

This example shows the full create-use-destroy lifecycle. Each step
is labeled with its purpose.

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::enums::*;
use vulkan_rs::vk::bitmasks::*;
use vulkan_rs::vk::handles::*;
use vulkan_rs::vk::Handle;
use vulkan_rs::Device;

unsafe fn buffer_lifecycle(device: &Device) {
    // ── Step 1: Describe what you want ──────────────────────────
    //
    // Every create call takes a CreateInfo struct. The builder
    // fills in sType automatically and provides a typed API
    // for each field.
    let buffer_info = BufferCreateInfo::builder()
        .size(1024)                           // 1 KiB
        .usage(BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(SharingMode::EXCLUSIVE);

    // ── Step 2: Create the object ───────────────────────────────
    //
    // The driver allocates the resource and returns a handle.
    // This can fail (out of memory, invalid parameters), so it
    // returns a Result.
    let buffer: Buffer = device
        .create_buffer(&buffer_info, None)
        .expect("Failed to create buffer");

    // The handle is just a number. You can copy it, compare it,
    // hash it, or check if it is null.
    assert!(!buffer.is_null());

    // ── Step 3: Use the handle ──────────────────────────────────
    //
    // You would normally bind memory to this buffer, then use
    // it in command buffer recording. For this example, we just
    // show that the handle is a lightweight Copy type.
    let buffer_copy = buffer;  // handles are Copy
    assert_eq!(buffer, buffer_copy);

    // ── Step 4: Destroy the object ──────────────────────────────
    //
    // You must destroy the buffer before destroying the Device
    // that created it. vulkan_rs does not track this for you.
    // There is no Drop implementation. You are responsible.
    device.destroy_buffer(buffer, None);

    // After this point, using `buffer` is undefined behavior.
    // Rust's type system does not prevent this, the handle is
    // still a valid Copy value. Vulkan's validation layers
    // will catch use-after-destroy if you enable them.
}
```

> *Before reading on: the code above calls `device.destroy_buffer(buffer, None)`.
> What do you think the second argument (`None`) is for? Hint: it relates
> to custom memory allocation, not GPU memory.*

### Objects that come from pools

CommandBuffers and DescriptorSets are not created individually. They
are *allocated in bulk* from a pool, and freed back to that pool (or
the entire pool is reset/destroyed at once):

```rust,ignore
use vulkan_rs::vk;
use vulkan_rs::vk::structs::*;
use vulkan_rs::vk::enums::*;
use vulkan_rs::vk::handles::*;
use vulkan_rs::vk::Handle;

// Pool-based lifecycle (simplified)
unsafe {
    // Create the pool (this is a normal create/destroy object).
    let pool_info = CommandPoolCreateInfo::builder()
        .queue_family_index(graphics_queue_family);
    let pool = device.create_command_pool(&pool_info, None)?;

    // Allocate command buffers FROM the pool.
    let alloc_info = CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(CommandBufferLevel::PRIMARY)
        .command_buffer_count(2);
    let mut command_buffers = [CommandBuffer::null(); 2];
    device.allocate_command_buffers(&alloc_info, command_buffers.as_mut_ptr())?;

    // Use command_buffers[0], command_buffers[1], ...

    // Option A: Free individual command buffers back to the pool.
    device.free_command_buffers(pool, &command_buffers);

    // Option B: Reset the entire pool (returns all buffers to initial state).
    device.reset_command_pool(pool, CommandPoolResetFlags::empty())?;

    // Destroy the pool (implicitly frees all remaining command buffers).
    device.destroy_command_pool(pool, None);
}
```

This pool pattern exists for performance: allocating and freeing
individual small objects is expensive, so Vulkan amortizes the cost
by batching them through pools.

### Objects that are enumerated, not created

PhysicalDevices and Queues are not created by you. They are discovered:

```rust,ignore
unsafe {
    // PhysicalDevices: the driver tells you what GPUs exist.
    let physical_devices = instance.enumerate_physical_devices()?;

    // Queues: retrieved from a Device after creation.
    let queue = device.get_device_queue(queue_family_index, 0);
}
```

You do not destroy enumerated objects. Their lifetime is tied to
their parent (PhysicalDevice lives as long as the Instance, Queue
lives as long as the Device).

## Formal reference

### The Handle trait

Every handle type in `vulkan_rs` implements the `Handle` trait:

```rust,ignore
pub trait Handle: Copy + Eq + Hash {
    type Repr;                       // usize or u64
    fn null() -> Self;               // the null handle (0)
    fn from_raw(raw: Self::Repr) -> Self;
    fn as_raw(self) -> Self::Repr;
    fn is_null(self) -> bool;
}
```

All handles also derive `Copy`, `Clone`, `PartialEq`, `Eq`, `Hash`,
`Default` (returns null), and `Debug` (prints the type name and hex value).

### Handle categories

| Category | Repr | Examples | Count |
|----------|------|----------|-------|
| Dispatchable | `usize` | Instance, PhysicalDevice, Device, CommandBuffer, Queue | 5 |
| Non-dispatchable | `u64` | Buffer, Image, Fence, Semaphore, Pipeline, ... | ~54 |

### Destruction rules

1. **You must destroy what you create.** `vulkan_rs` has no `Drop`
   implementations on handles. This is deliberate: automatic destruction
   would require tracking creation order, reference counting, and
   deferred destruction (the GPU might still be using the object). That
   complexity belongs in your application, not in the bindings.

2. **Destroy children before parents.** The tree above defines the order.
   Validation layers will warn you if you get it wrong.

3. **The GPU must be done with an object before you destroy it.** If a
   command buffer references a Buffer that you then destroy, the GPU
   will read freed memory. Use fences or `device_wait_idle()` to
   ensure GPU work has completed.

4. **Pool destruction frees all children.** Destroying a CommandPool
   implicitly frees all CommandBuffers allocated from it. Same for
   DescriptorPool and DescriptorSets.

5. **Enumerated objects are not destroyed.** PhysicalDevice and Queue
   handles are valid for the lifetime of their parent.

### Interop: from_raw_parts

If another system creates Vulkan objects for you (OpenXR, a C library,
a test harness), you can wrap them:

```rust,ignore
// Wrap an externally-created Instance.
let instance = unsafe {
    Instance::from_raw_parts(raw_instance_handle, get_instance_proc_addr_fn)
};

// Wrap an externally-created Device.
let device = unsafe {
    Device::from_raw_parts(raw_device_handle, get_device_proc_addr_fn)
};
```

The wrapped objects load all function pointers from the provided
`get_*_proc_addr` function, so they work identically to objects
created through `Entry::create_instance`.

### API reference links

- [`Handle` trait](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/trait.Handle.html)
- [`Instance`](https://docs.rs/vulkan-rs/latest/vulkan_rs/struct.Instance.html)
- [`Device`](https://docs.rs/vulkan-rs/latest/vulkan_rs/struct.Device.html)
- [`Buffer`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.Buffer.html)
- [Vulkan spec: Object Model](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#fundamentals-objectmodel-overview)

## Key takeaways

- Vulkan handles are opaque numbers, not pointers to inspectable structs.
- Handles form a parent-child tree. Create bottom-up, destroy top-down.
- Most objects follow create → use → destroy. Pools and enumerated objects
  are the two exceptions.
- `vulkan_rs` gives you `Copy` handles with no `Drop`. You manage lifetimes.
  Validation layers are your safety net during development.
