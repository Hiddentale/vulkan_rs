# Design Decisions & Safety Model

This page explains the major design decisions in `vulkan_rs` and why they
were made. Each section addresses a common "why not do it the other way?"
question.

## Why two crates?

`vulkan_rs` is split into two crates with distinct roles:

- **`vulkan-rs-sys`** is machine-generated from `vk.xml`. It contains ~40,000 lines
  of `#[repr(C)]` structs, `#[repr(transparent)]` enum newtypes, bitmask
  types, handle types, and function pointer typedefs. It is `#![no_std]`.
- **`vulkan-rs`** is hand-written. It provides `Entry`, `Instance`, `Device`,
  command loading, builders, surface helpers, and the error types.

Users depend on `vulkan-rs` and access raw types via `vulkan_rs::vk::*`.

This separation exists for three reasons:

1. **Build speed.** Regenerating `vulkan-rs-sys` only happens when a new Vulkan
   spec version arrives. Day-to-day development in `vulkan-rs` does not
   trigger a rebuild of 40k lines of generated code.
2. **Reviewability.** Generated code is validated by the generator's test
   suite, not by human review. Hand-written code gets normal review. Mixing
   them in one crate blurs that boundary.
3. **`no_std` compatibility.** `vulkan-rs-sys` has zero dependencies and can be
   used in environments without `std`. `vulkan-rs` requires `std` for
   library loading and allocation.

## Why inherent methods instead of traits?

All Vulkan commands are inherent methods on `Device` or `Instance`:

```rust,ignore
use vulkan_rs::vk;

// No trait import needed, just call the method.
let buffer = unsafe { device.create_buffer(&info, None) }?;
```

Some Vulkan wrappers split commands across extension traits (e.g.
`KhrSwapchainExtension`). This forces callers to import the right trait
before calling the method, and IDE autocomplete only works when the trait
is already in scope.

With inherent methods, every command appears in autocomplete on `Device`
immediately, and there is nothing to import.

## Why complete command loading?

When `Device` or `Instance` is created, `vulkan_rs` loads **every** function
pointer from every enabled extension in a single pass. Some wrappers require
callers to explicitly request which extension command tables to load.

Complete loading avoids that bookkeeping. The cost is negligible: loading a
few hundred function pointers takes microseconds at startup, and the
per-pointer memory cost is one `Option<fn>` each.

## Why `from_raw_parts`?

Both `Instance` and `Device` provide an `unsafe fn from_raw_parts`
constructor that wraps an externally-owned Vulkan handle:

```rust,ignore
use vulkan_rs::Device;

let device = unsafe {
    Device::from_raw_parts(raw_vk_device, Some(get_device_proc_addr_fn))
};
```

This exists for three use cases:

1. **OpenXR interop.** The XR runtime creates the `VkInstance` and
   `VkDevice`. Your code receives raw handles and needs to wrap them.
2. **Middleware.** Profiling layers and debug tools may hand you raw handles.
3. **Testing.** Unit tests can construct wrapper objects without a real GPU.

## Why no Drop on handles?

`Instance` and `Device` do not implement `Drop`. Destruction is explicit:

```rust,ignore
use vulkan_rs::vk;

unsafe { device.destroy_device(None) };
```

Automatic destruction via `Drop` is tempting, but breaks in several
real scenarios:

- **`from_raw_parts` and shared ownership.** If two wrappers hold the same
  raw handle (e.g. your code and an OpenXR runtime), a `Drop` impl would
  double-destroy it.
- **GPU-async lifetimes.** The GPU may still be using resources when Rust
  drops a handle. Correct destruction requires calling `device_wait_idle`
  or using fences first. A `Drop` impl cannot know when the GPU is done.
- **Destruction order.** Vulkan objects have strict parent-child destruction
  ordering. Rust's drop order (reverse declaration order within a scope) may
  not match what Vulkan requires.

Explicit destruction makes the caller responsible, which matches Vulkan's
own model.

## Why builders Deref to the inner struct?

Every builder dereferences to its inner `vk::*` struct:

```rust,ignore
use vulkan_rs::vk;
use vk::structs::*;
use vk::bitmasks::*;

let info = BufferCreateInfo::builder()
    .size(1024)
    .usage(BufferUsageFlags::VERTEX_BUFFER);

// Pass the builder directly where a &BufferCreateInfo is expected.
let buffer = unsafe { device.create_buffer(&info, None) }?;
```

Because `BufferCreateInfoBuilder` implements `Deref<Target = BufferCreateInfo>`,
there is no `.build()` call. The builder *is* the struct, with a convenient
setter API on top. This means you can pass a builder reference anywhere a
struct reference is expected.

## Why `#[repr(transparent)]` newtypes for enums?

Vulkan "enums" are integer constants, not closed sets. Drivers and extensions
can return values that did not exist when your code was compiled. A Rust
`enum` with unknown discriminants is instant undefined behavior.

Instead, `vulkan-rs-sys` represents each Vulkan enum as a `#[repr(transparent)]`
newtype around `i32`:

```rust,ignore
use vulkan_rs::vk;
use vk::enums::*;

#[repr(transparent)]
pub struct Format(i32);

impl Format {
    pub const UNDEFINED: Self = Self(0);
    pub const R8G8B8A8_UNORM: Self = Self(37);
    // ... hundreds more
}
```

Unknown values are perfectly legal, they just lack a named constant.
Pattern matching uses associated constants, and the compiler does not
assume the set is exhaustive.

## The safety model

**All Vulkan command wrappers are `unsafe fn`.** The caller is responsible
for meeting every precondition the Vulkan spec defines: valid handles,
correct synchronization, matching lifetimes, and so on.

`vulkan_rs` does not attempt to encode Vulkan's safety rules in the Rust
type system. The spec is too large and too nuanced for compile-time
enforcement to be practical without severe ergonomic cost.

Instead, the safety strategy is:

1. **Validation layers during development.** Enable `VK_LAYER_KHRONOS_validation`
   in debug builds. The validation layer catches spec violations, use-after-free,
   missing synchronization, and more. It is the primary safety net.
2. **Type-safe newtypes.** You cannot accidentally pass a `Buffer` where a
   `Pipeline` is expected. This catches a class of handle mixups at compile time.
3. **Builder `push_next` with marker traits.** The `push_next` method on
   builders is generic over an `Extends*` marker trait, so only structs that
   the spec allows in a given pNext chain can be appended.
4. **Panic on missing function pointers.** If you call a command from an
   extension that was not enabled, the stub panics with a descriptive message
   (e.g. `"VK_KHR_surface not loaded"`). This catches misconfiguration early.

## What the generator handles vs what is hand-written

| Generated (`vulkan-rs-sys`)                  | Hand-written (`vulkan-rs`)                |
|----------------------------------------|-------------------------------------------|
| `#[repr(C)]` struct definitions        | `Entry`, `Instance`, `Device` wrappers    |
| `#[repr(transparent)]` enum newtypes   | Command loading and dispatch tables       |
| Bitmask types and flag constants       | `from_raw_parts` constructors             |
| Handle newtypes                        | Error types (`VkResult`, `LoadError`)     |
| Function pointer typedefs              | Surface creation (`SurfaceError`)         |
| Builder structs with `Deref`           | SPIR-V bytecode loading                   |
| `push_next` methods + `Extends*` traits| Version parsing                           |
| Wrapper methods on `Device`/`Instance` | `Loader` trait and library loading        |
