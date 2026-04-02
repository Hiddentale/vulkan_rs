# Validation Layers & Debugging

## Motivation

Vulkan does almost no error checking at runtime, calling a function
incorrectly is undefined behavior, not an error message. This is fast
but makes debugging brutal. A typo in a pipeline barrier's access mask
won't crash immediately; it will cause a subtle rendering glitch three
frames later on one specific GPU.

Validation layers are optional middleware that intercepts every Vulkan
call and checks it against the spec. They catch invalid usage, report
synchronization hazards, and point you to the exact spec section that
explains what went wrong. You should *always* enable them during
development.

## Intuition

### The strict code reviewer

Validation layers are a strict code reviewer sitting between your
application and the driver. Every API call passes through the reviewer
first. In development, the reviewer catches your mistakes before they
reach the driver. In production, you remove the reviewer and calls go
straight through.

```text
Your app ──> Validation Layer ──> Vulkan Driver ──> GPU
              │
              │ "ERROR: Buffer 0x42 was not created with
              │  TRANSFER_DST usage, but you're using it
              │  as a copy destination. See spec section 7.4."
              v
            Callback (your code logs or prints this)
```

Without validation layers:

```text
Your app ──────────────────────> Vulkan Driver ──> GPU
                                                    │
                                                    │ (undefined behavior,
                                                    │  maybe works, maybe
                                                    │  corrupts memory,
                                                    │  maybe crashes later)
```

> *Before reading on: why do you think Vulkan chose to make error
> checking optional instead of always-on?*
>
> Answer: Performance. Validation checking every API call adds
> measurable overhead (sometimes 2-5x slower). For a shipped game
> running at 60fps, that cost is unacceptable. By making validation
> optional, development builds get thorough checking while release
> builds get maximum performance.

## Worked example: enabling validation with a debug messenger

### Step 1: Enable the validation layer at instance creation

```rust,ignore
use std::ffi::CStr;
use vulkan_rs::vk;
use vk::structs::*;

// The standard validation layer name.
let validation_layer = c"VK_LAYER_KHRONOS_validation";
let layer_names = [validation_layer.as_ptr()];

// The debug utils extension lets us receive callbacks.
let extension_names = [
    c"VK_EXT_debug_utils".as_ptr(),
];

let instance_info = InstanceCreateInfo::builder()
    .enabled_layer_names(&layer_names)
    .enabled_extension_names(&extension_names);

let instance = unsafe { entry.create_instance(&instance_info, None)? };
```

### Step 2: Set up a debug messenger

The debug messenger calls your function whenever validation finds a
problem.

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::bitmasks::*;

// This callback receives validation messages.
// The signature must match PFN_vkDebugUtilsMessengerCallbackEXT.
unsafe extern "system" fn debug_callback(
    severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut core::ffi::c_void,
) -> u32 {
    let message = if !callback_data.is_null() {
        let data = &*callback_data;
        if !data.p_message.is_null() {
            CStr::from_ptr(data.p_message).to_string_lossy()
        } else {
            std::borrow::Cow::Borrowed("(no message)")
        }
    } else {
        std::borrow::Cow::Borrowed("(no callback data)")
    };

    if severity & DebugUtilsMessageSeverityFlagsEXT::ERROR
        != DebugUtilsMessageSeverityFlagsEXT::empty()
    {
        eprintln!("[VULKAN ERROR] {message}");
    } else if severity & DebugUtilsMessageSeverityFlagsEXT::WARNING
        != DebugUtilsMessageSeverityFlagsEXT::empty()
    {
        eprintln!("[VULKAN WARNING] {message}");
    }

    0 // returning 1 would abort the Vulkan call that triggered this
}
```

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::bitmasks::*;

// Create the messenger.
let messenger_info = DebugUtilsMessengerCreateInfoEXT::builder()
    .message_severity(
        DebugUtilsMessageSeverityFlagsEXT::WARNING
        | DebugUtilsMessageSeverityFlagsEXT::ERROR,
    )
    .message_type(
        DebugUtilsMessageTypeFlagsEXT::GENERAL
        | DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
    )
    .pfn_user_callback(Some(debug_callback));

let messenger = unsafe {
    instance.create_debug_utils_messenger_ext(&messenger_info, None)?
};
```

### Step 3: Trigger an error (intentionally)

To verify validation is working, do something wrong on purpose:

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::enums::*;
use vk::bitmasks::*;

// Create a buffer without TRANSFER_DST usage, then try to copy into it.
let bad_buffer_info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlags::VERTEX_BUFFER)  // no TRANSFER_DST!
    .sharing_mode(SharingMode::EXCLUSIVE);

let bad_buffer = unsafe { device.create_buffer(&bad_buffer_info, None)? };

// Recording a copy to this buffer will produce a validation error:
// "vkCmdCopyBuffer: dstBuffer was not created with VK_BUFFER_USAGE_TRANSFER_DST_BIT"
```

### Step 4: Clean up

```rust,ignore
use vulkan_rs::vk;

// Destroy the messenger before destroying the instance.
unsafe {
    instance.destroy_debug_utils_messenger_ext(messenger, None);
};
```

## Message severity levels

| Severity | Meaning | Action |
|----------|---------|--------|
| `VERBOSE` | Diagnostic noise (loader info, layer status) | Usually filtered out |
| `INFO` | Informational (resource creation, state changes) | Useful for deep debugging |
| `WARNING` | Potential problem (suboptimal usage, deprecated behavior) | Investigate |
| `ERROR` | Spec violation (undefined behavior if ignored) | Fix immediately |

Filter severity in the messenger creation to control verbosity. Most
applications enable `WARNING | ERROR` and only enable `VERBOSE | INFO`
when debugging specific issues.

## Message types

| Type | What it checks |
|------|---------------|
| `GENERAL` | General events (loader, layer lifecycle) |
| `VALIDATION` | Spec violations (the most important type) |
| `PERFORMANCE` | Suboptimal API usage that may hurt performance |
| `DEVICE_ADDRESS_BINDING` | Buffer device address binding events |

## Catching errors during instance creation

There is a bootstrap problem: you need an instance to create a debug
messenger, but errors can occur *during* instance creation. The
solution: chain the messenger create info into the instance create info
via pNext. The validation layer will use it for messages during
`create_instance`:

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::bitmasks::*;

let mut debug_info = DebugUtilsMessengerCreateInfoEXT::builder()
    .message_severity(
        DebugUtilsMessageSeverityFlagsEXT::WARNING
        | DebugUtilsMessageSeverityFlagsEXT::ERROR,
    )
    .message_type(
        DebugUtilsMessageTypeFlagsEXT::GENERAL
        | DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
    )
    .pfn_user_callback(Some(debug_callback));

// Chain into instance creation via pNext.
// DebugUtilsMessengerCreateInfoEXT implements ExtendsInstanceCreateInfo.
let instance_info = InstanceCreateInfo::builder()
    .enabled_layer_names(&layer_names)
    .enabled_extension_names(&extension_names)
    .push_next(&mut debug_info);

// Validation errors during create_instance will now trigger the callback.
let instance = unsafe { entry.create_instance(&instance_info, None)? };

// After instance creation, create a persistent messenger for the
// rest of the application's lifetime.
let messenger = unsafe {
    instance.create_debug_utils_messenger_ext(&debug_info, None)?
};
```

This is a practical example of pNext in action (see
[The pNext Extension Chain](pnext.md)).

## Common validation errors and what they mean

| Error message (abbreviated) | Cause | Fix |
|---------------------------|-------|-----|
| "not created with ... usage" | Resource missing a usage flag | Add the required usage flag at creation |
| "layout is UNDEFINED but expected ..." | Image in wrong layout | Add a pipeline barrier to transition |
| "access mask ... not supported by stage ..." | Access mask doesn't match pipeline stage | Check the [barrier recipes](synchronization.md) table |
| "must not be in RECORDING state" | Submitting a command buffer that wasn't ended | Call `end_command_buffer` before submitting |
| "is still in use by the GPU" | Destroying an object the GPU is using | Wait for the fence before destroying |
| "extension not enabled" | Using an extension feature without enabling it | Add the extension to instance/device creation |

## Performance impact

Validation layers add significant overhead:

- **CPU time:** Every API call is checked against the spec. Expect 2-5x
  slower CPU-side Vulkan calls.
- **Memory:** The layer tracks all objects and their state.
- **GPU time:** Minimal, but synchronization validation may serialize
  GPU work.

Always disable validation in release builds. A common pattern:

```rust,ignore
let enable_validation = cfg!(debug_assertions);

let layer_names: Vec<*const i8> = if enable_validation {
    vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
} else {
    vec![]
};
```

## Formal reference

### Key types

| Type | Purpose |
|------|---------|
| `DebugUtilsMessengerEXT` | Handle to the debug messenger |
| `DebugUtilsMessengerCreateInfoEXT` | Configuration: severity filter, type filter, callback |
| `DebugUtilsMessageSeverityFlagsEXT` | Severity bitmask (VERBOSE, INFO, WARNING, ERROR) |
| `DebugUtilsMessageTypeFlagsEXT` | Type bitmask (GENERAL, VALIDATION, PERFORMANCE) |

### Required extension

The debug messenger requires the `VK_EXT_debug_utils` instance
extension. Enable it with `c"VK_EXT_debug_utils"` as a C string literal.

### Destruction order

1. Destroy the debug messenger before destroying the instance.
2. The pNext-chained messenger (for instance creation) is temporary
   and does not need separate destruction.

### API reference links

- [`DebugUtilsMessengerEXT`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DebugUtilsMessengerEXT.html)
- [`DebugUtilsMessengerCreateInfoEXT`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DebugUtilsMessengerCreateInfoEXT.html)
- [`DebugUtilsMessageSeverityFlagsEXT`](https://docs.rs/vulkan-rs/latest/vulkan_rs/vk/struct.DebugUtilsMessageSeverityFlagsEXT.html)
- [Vulkan spec: Debugging](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#debugging)

## Key takeaways

- Always enable validation layers during development. They catch
  undefined behavior that would otherwise silently corrupt rendering.
- Set up a debug messenger callback to receive errors in your code.
  Don't rely on console output, some platforms don't have one.
- Chain `DebugUtilsMessengerCreateInfoEXT` into `InstanceCreateInfo`
  via pNext to catch errors during instance creation.
- Filter by severity (WARNING + ERROR) and type (VALIDATION + PERFORMANCE)
  for the best signal-to-noise ratio.
- Disable validation in release builds. The overhead is significant.
