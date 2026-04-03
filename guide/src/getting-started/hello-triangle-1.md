# Hello Triangle, Part 1: Instance & Device

This is the first part of a four-part tutorial that builds a complete
Vulkan application from scratch. By the end of part 4, you will have a
colored triangle on screen. By the end of *this* part, you will have
a working connection to your GPU.

**What we build in this part:**

```text
Load Vulkan ──> Create Instance ──> Pick a GPU ──> Create Device ──> Get a Queue
```

Each step depends on the previous one. We will take them one at a time,
with an explanation of *why* each step exists before the code.

## Prerequisites

- [Install vulkan_rust and the Vulkan SDK](installation.md)
- A working Rust toolchain (`cargo build` succeeds)
- A system with a Vulkan-capable GPU

## Create the project

```bash
cargo new hello-triangle
cd hello-triangle
```

Add `vulkan-rust` to your `Cargo.toml`:

```toml
[dependencies]
vulkan-rust = "0.9"
```

## Step 1: Load the Vulkan library

Before you can call any Vulkan function, you must load the Vulkan shared
library (`vulkan-1.dll` on Windows, `libvulkan.so` on Linux,
`libvulkan.dylib` on macOS). This library is the **loader**, the
entry point that routes your calls to the correct GPU driver.

```rust,no_run
use vulkan_rust::{Entry, LibloadingLoader};

fn main() {
    // Load the Vulkan shared library from the system.
    // This can fail if the Vulkan SDK is not installed.
    let loader = LibloadingLoader::new()
        .expect("Failed to find Vulkan library");

    // Create the Entry, which resolves the bootstrap function pointers
    // (vkGetInstanceProcAddr, vkGetDeviceProcAddr).
    let entry = unsafe { Entry::new(loader) }
        .expect("Failed to load Vulkan entry points");

    // Verify: query the highest Vulkan version the driver supports.
    let version = entry.version().expect("Failed to query Vulkan version");
    println!("Vulkan {}.{}.{}", version.major, version.minor, version.patch);
}
```

Run this with `cargo run`. If you see output like `Vulkan 1.3.280`, your
setup is working.

> *Why is this `unsafe`?* Loading a shared library and calling its
> functions through raw pointers is inherently unsafe. The compiler
> cannot verify that the library is valid or that the function pointers
> it returns are correct. This is the only `unsafe` we need to
> *understand* right now; the rest follow the same pattern.

## Step 2: Create a Vulkan Instance

An Instance is your application's connection to the Vulkan runtime. It
loads the driver, enables validation layers, and provides access to
the physical GPUs on the system.

Think of it as opening a session: "I am application X, I want to use
Vulkan version Y, please give me access."

```rust,ignore
use vulkan_rust::vk;
use vk::*;

// ── Describe your application ──────────────────────────────────
//
// ApplicationInfo tells the driver who you are. This is optional
// but helps driver vendors optimize for known applications.
let app_info = ApplicationInfo::builder()
    .application_name(c"Hello Triangle")
    .application_version(1)
    .engine_name(c"No Engine")
    .engine_version(1)
    .api_version(1 << 22);  // Vulkan 1.0

// ── Describe what you need ─────────────────────────────────────
//
// No layers or extensions yet. We will add validation layers and
// surface extensions in later parts.
let create_info = InstanceCreateInfo::builder()
    .application_info(&app_info);

// ── Create the instance ────────────────────────────────────────
let instance = unsafe { entry.create_instance(&create_info, None) }
    .expect("Failed to create Vulkan instance");

println!("Instance created successfully");
```

> *Before reading on: why do you think the Instance takes an
> `api_version` field? What would happen if you requested a version
> the driver doesn't support?*

The `api_version` tells the driver the highest Vulkan version your
application is written against. If the driver supports that version or
higher, it succeeds. If you request 1.3 on a 1.0-only driver, instance
creation fails with `ERROR_INCOMPATIBLE_DRIVER`.

## Step 3: Pick a physical device (GPU)

A system can have multiple GPUs: a discrete NVIDIA/AMD card, an
integrated Intel GPU, or even a software renderer. You must choose one.

```rust,ignore
use vk::PhysicalDeviceType;

// ── Enumerate GPUs ─────────────────────────────────────────────
let physical_devices = unsafe { instance.enumerate_physical_devices() }
    .expect("Failed to enumerate GPUs");

println!("Found {} GPU(s):", physical_devices.len());

// ── Inspect each one ───────────────────────────────────────────
for (i, &pd) in physical_devices.iter().enumerate() {
    let props = unsafe { instance.get_physical_device_properties(pd) };

    // The device name is a null-terminated C string in a fixed-size array.
    let name_bytes: Vec<u8> = props.device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    let name = String::from_utf8_lossy(&name_bytes);

    let device_type = match props.device_type {
        PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        PhysicalDeviceType::CPU => "CPU (software)",
        _ => "Other",
    };

    println!("  [{}] {} ({})", i, name, device_type);
}

// ── Pick the first GPU ─────────────────────────────────────────
//
// A real application would score GPUs by capability (discrete >
// integrated, required features, memory size). For this tutorial,
// the first one is fine.
let physical_device = physical_devices[0];
```

> *Before reading on: the code above uses `get_physical_device_properties`
> to read the GPU name and type. What other information do you think
> the driver exposes about each physical device?*

The `PhysicalDeviceProperties` struct also contains the driver version,
the Vulkan API version the device supports, and `limits`, a struct
with hundreds of fields describing maximum texture sizes, buffer
alignments, and other hardware limits.

## Step 4: Find a queue family that supports graphics

The GPU exposes **queues**, which are the endpoints where you submit
work. Queues are grouped into **families**, where each family supports
a specific set of operations (graphics, compute, transfer, etc.).

We need a queue family that supports graphics operations.

```rust,ignore
use vk::QueueFlags;

// ── Query queue families ───────────────────────────────────────
let queue_families = unsafe {
    instance.get_physical_device_queue_family_properties(physical_device)
};

// ── Find one that supports graphics ────────────────────────────
let graphics_family_index = queue_families
    .iter()
    .enumerate()
    .find(|(_, family)| {
        family.queue_flags & QueueFlags::GRAPHICS
            != QueueFlags::empty()
    })
    .map(|(index, _)| index as u32)
    .expect("No graphics queue family found");

println!("Using queue family {} for graphics", graphics_family_index);
```

Queue families are identified by their index in the array. We will pass
this index to device creation (to request a queue from that family) and
to many other calls throughout the application.

## Step 5: Create a logical Device

A Device is your interface to one physical GPU. It loads all the
device-level function pointers and provides the methods you will use
for the rest of the application: creating buffers, recording commands,
submitting work.

Creating a Device also creates the queues you requested.

```rust,ignore
use vk::*;

// ── Request one queue from the graphics family ─────────────────
let queue_priority = 1.0_f32;

let queue_info = DeviceQueueCreateInfo::builder()
    .queue_family_index(graphics_family_index)
    .queue_priorities(std::slice::from_ref(&queue_priority));

// ── Create the device ──────────────────────────────────────────
//
// No extensions or features yet. We will add the swapchain
// extension in Part 2.
let device_info = DeviceCreateInfo::builder()
    .queue_create_infos(std::slice::from_ref(&queue_info));

let device = unsafe {
    instance.create_device(physical_device, &device_info, None)
}
.expect("Failed to create logical device");

println!("Device created successfully");
```

> *Before reading on: we requested a queue with priority `1.0`. What
> do you think the priority controls?*

Queue priority is a hint to the driver about how to schedule work when
multiple queues compete for GPU resources. `1.0` is the highest
priority. Most applications use a single queue and set it to `1.0`.
The actual effect is driver-dependent.

## Step 6: Get a queue handle

The Device created our queues internally. We retrieve handles to them
with `get_device_queue`.

```rust,ignore
// ── Retrieve the graphics queue ────────────────────────────────
//
// Queue family index: the family we chose above.
// Queue index: 0, because we only requested 1 queue from this family.
let graphics_queue = unsafe {
    device.get_device_queue(graphics_family_index, 0)
};

println!("Graphics queue ready");
```

The queue handle is not created or destroyed by you. It is owned by the
Device and valid for the Device's lifetime. (See
[The Vulkan Object Model](../concepts/object-model.md) for the
distinction between created, allocated, and enumerated objects.)

## Step 7: Clean up

Vulkan requires explicit destruction in reverse creation order.
`vulkan_rust` has no `Drop` implementations on purpose, so you must
call the destroy methods yourself.

```rust,ignore
// ── Destroy in reverse order ───────────────────────────────────
//
// Queue handles are owned by the Device, no destroy needed.
// Device must be destroyed before Instance.
// Instance must be destroyed last.
unsafe {
    device.destroy_device(None);
    instance.destroy_instance(None);
}

println!("Cleaned up successfully");
```

## Putting it all together

Here is the complete program. Copy this into `src/main.rs` and run it
with `cargo run`.

```rust,no_run
use vulkan_rust::{Entry, LibloadingLoader};
use vulkan_rust::vk;
use vk::*;

fn main() {
    // ── Step 1: Load Vulkan ────────────────────────────────────
    let loader = LibloadingLoader::new()
        .expect("Vulkan library not found");
    let entry = unsafe { Entry::new(loader) }
        .expect("Failed to load Vulkan");

    let version = entry.version().expect("Failed to query version");
    println!("Vulkan {}.{}.{}", version.major, version.minor, version.patch);

    // ── Step 2: Create Instance ────────────────────────────────
    let app_info = ApplicationInfo::builder()
        .application_name(c"Hello Triangle")
        .application_version(1)
        .engine_name(c"No Engine")
        .engine_version(1)
        .api_version(1 << 22);  // Vulkan 1.0

    let create_info = InstanceCreateInfo::builder()
        .application_info(&app_info);

    let instance = unsafe { entry.create_instance(&create_info, None) }
        .expect("Failed to create instance");

    // ── Step 3: Pick a GPU ─────────────────────────────────────
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .expect("Failed to enumerate GPUs");

    let physical_device = physical_devices[0];

    let props = unsafe {
        instance.get_physical_device_properties(physical_device)
    };
    let name_bytes: Vec<u8> = props.device_name
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    println!("GPU: {}", String::from_utf8_lossy(&name_bytes));

    // ── Step 4: Find a graphics queue family ───────────────────
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    let graphics_family_index = queue_families
        .iter()
        .enumerate()
        .find(|(_, family)| {
            family.queue_flags & QueueFlags::GRAPHICS
                != QueueFlags::empty()
        })
        .map(|(index, _)| index as u32)
        .expect("No graphics queue family found");

    // ── Step 5: Create Device ──────────────────────────────────
    let queue_priority = 1.0_f32;
    let queue_info = DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_family_index)
        .queue_priorities(std::slice::from_ref(&queue_priority));

    let device_info = DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info));

    let device = unsafe {
        instance.create_device(physical_device, &device_info, None)
    }
    .expect("Failed to create device");

    // ── Step 6: Get the graphics queue ─────────────────────────
    let _graphics_queue = unsafe {
        device.get_device_queue(graphics_family_index, 0)
    };

    println!("Vulkan initialized successfully!");
    println!("Ready for Part 2: Swapchain & Surface");

    // ── Step 7: Clean up ───────────────────────────────────────
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
```

Expected output:

```text
Vulkan 1.3.280
GPU: NVIDIA GeForce RTX 4070
Vulkan initialized successfully!
Ready for Part 2: Swapchain & Surface
```

(Your version number and GPU name will differ.)

## What we learned

This part covered the Vulkan initialization sequence:

| Step | What | Why |
|------|------|-----|
| Load library | `LibloadingLoader::new()` + `Entry::new()` | Get access to Vulkan function pointers |
| Create Instance | `entry.create_instance()` | Open a session with the Vulkan driver |
| Pick GPU | `enumerate_physical_devices()` + `get_physical_device_properties()` | Choose which hardware to use |
| Find queue family | `get_physical_device_queue_family_properties()` | Find a queue that supports graphics |
| Create Device | `instance.create_device()` | Get a logical interface to the GPU |
| Get queue | `device.get_device_queue()` | Get the submission endpoint |

Every Vulkan application does these steps. They are the foundation that
everything else builds on.

## What we skipped (and will add later)

- **Validation layers** (Part 2), catch API misuse during development.
  See [Validation Layers](../concepts/validation.md) for the concept.
- **Surface and swapchain** (Part 2), connect to a window so we can
  display pixels.
- **Extensions**, we will enable `VK_KHR_swapchain` and surface
  extensions in Part 2.

## Exercises

1. **Print all GPUs.** Modify the program to print every physical device
   with its name and type, not just the first one.
2. **Print all queue families.** For the chosen GPU, print every queue
   family with its flags (GRAPHICS, COMPUTE, TRANSFER) and queue count.
3. **Choose discrete over integrated.** Modify the GPU selection to
   prefer a discrete GPU when one is available.

## Next

[Part 2: Swapchain & Surface](hello-triangle-2.md) adds a window,
creates a swapchain, and introduces validation layers.
