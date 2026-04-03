# Error Handling Philosophy

This page explains how `vulkan_rust` maps Vulkan's C-style error model into
idiomatic Rust, and where the boundaries between error types lie.

## Vulkan's error model

Every Vulkan command that can fail returns a `VkResult`, which is a plain
`int32_t`. The spec defines named constants for it:

- **Success codes** are non-negative: `VK_SUCCESS` (0), `VK_INCOMPLETE` (5),
  `VK_SUBOPTIMAL_KHR` (1000001003), and a few others.
- **Error codes** are negative: `VK_ERROR_OUT_OF_HOST_MEMORY` (-1),
  `VK_ERROR_DEVICE_LOST` (-2), etc.

There is no exception system, no errno, no callback. The caller checks the
return value after every call.

## The `VkResult<T>` type alias

`vulkan-rust` defines a single result type for all Vulkan command wrappers:

```rust,ignore
use vulkan_rust::vk;

pub type VkResult<T> = std::result::Result<T, vk::Result>;
```

Here `vk::Result` is the `#[repr(transparent)]` i32 newtype from `vulkan-rust-sys`.
The `Err` variant holds any negative value. The `Ok` variant holds the
command's output (a handle, a vector of properties, or just `()`).

A helper function performs the conversion:

```rust,ignore
use vulkan_rust::vk;

pub(crate) fn check(result: vk::Result) -> VkResult<()> {
    if result.as_raw() >= 0 {
        Ok(())
    } else {
        Err(result)
    }
}
```

This means all non-negative codes, including `INCOMPLETE` and `SUBOPTIMAL`,
are treated as success by default.

## Success codes that are not SUCCESS

Some Vulkan commands return positive success codes that carry meaning:

- **`INCOMPLETE`** from enumeration commands means the output buffer was
  too small. `vulkan-rust`'s two-call helpers handle this internally by
  querying the count first, so callers rarely see it.
- **`SUBOPTIMAL_KHR`** from `vkAcquireNextImageKHR` means the swapchain
  still works but no longer matches the surface optimally. You should
  recreate the swapchain, but the current frame is still valid.

Because `check` maps all non-negative codes to `Ok(())`, these success
codes do not propagate as errors. Wrapper methods that need to distinguish
them (e.g. swapchain acquisition) inspect the raw code explicitly after
the `check` call.

## `LoadError` for library loading

Before any Vulkan command runs, the shared library (`vulkan-1.dll`,
`libvulkan.so`) must be loaded and `vkGetInstanceProcAddr` resolved.
Failures here are not Vulkan API errors, they mean the Vulkan runtime
is not available at all.

`LoadError` captures these:

```rust,ignore
use vulkan_rust::vk;

pub enum LoadError {
    /// The Vulkan shared library could not be found or opened.
    Library(libloading::Error),
    /// vkGetInstanceProcAddr could not be resolved from the library.
    MissingEntryPoint,
}
```

`LoadError` implements `std::error::Error` and is returned from
`Entry::new`. It is entirely separate from `vk::Result`.

## `SurfaceError` for surface creation

Creating a window surface involves platform-specific logic and
`raw-window-handle` integration. Three distinct failure modes exist:

```rust,ignore
use vulkan_rust::vk;

pub enum SurfaceError {
    /// The display/window handle combination is not supported.
    UnsupportedPlatform,
    /// raw-window-handle returned an error.
    HandleError(raw_window_handle::HandleError),
    /// Vulkan error from the surface creation call.
    Vulkan(vk::Result),
}
```

`SurfaceError` unifies platform detection failures, handle errors, and
the underlying Vulkan error into one type, so callers of
`Instance::create_surface` have a single `Result` to handle.

## When vulkan_rust panics

Panics are reserved for **programmer mistakes**, never for runtime failures
that a correct program could encounter:

- **Calling an unloaded function pointer.** If you call a command from an
  extension that was not enabled at instance or device creation, the function
  pointer is `None`. The generated wrapper calls `.expect()` with a message
  like `"VK_KHR_surface not loaded"`. This is a configuration error, not a
  recoverable failure.
- **Internal invariant violations.** These should never happen. If they do,
  a panic with a descriptive message is the right response.

Vulkan runtime errors (out of memory, device lost, surface lost) are always
returned as `Err(vk::Result)`, never panicked.

## The standard pattern

Most application code follows the same pattern: call the command, propagate
errors with `?`, handle them at the boundary.

```rust,ignore
use vulkan_rust::vk;
use vulkan_rust::Device;
use vk::*;

unsafe fn create_pipeline(
    device: &Device,
    layout: PipelineLayout,
    render_pass: RenderPass,
    // ...
) -> VkResult<Pipeline> {
    let shader = device.create_shader_module(&shader_info, None)?;
    let pipeline = device.create_graphics_pipelines(
        PipelineCache::null(),
        &[pipeline_info],
        None,
    )?[0];
    device.destroy_shader_module(shader, None);
    Ok(pipeline)
}
```

Individual commands propagate errors upward. The top-level caller (your
main loop or initialization function) decides whether to retry, fall back,
or exit.

## Validation layers vs error codes

These are complementary, not overlapping:

| Concern | Mechanism |
|---------|-----------|
| Spec violations (wrong usage, missing sync) | Validation layers (`VK_LAYER_KHRONOS_validation`) |
| Recoverable runtime failures (OOM, device lost) | `vk::Result` error codes via `VkResult<T>` |
| Missing Vulkan runtime | `LoadError` |
| Platform surface issues | `SurfaceError` |
| Programmer misconfiguration (extension not enabled) | Panic |

Validation layers are a development-time tool. They intercept every Vulkan
call, check it against the spec, and report violations via debug callbacks.
They have significant overhead and are typically disabled in release builds.

Error codes are a production-time mechanism. They report conditions the
application can respond to: allocate less memory, recreate the swapchain,
or shut down gracefully.

A well-structured `vulkan_rust` application uses both: validation layers to
catch bugs during development, error propagation to handle failures in
production.
