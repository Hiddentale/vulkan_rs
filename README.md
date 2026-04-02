# vulkan-rs

<img align="right" width="20%" src="logo.png">

[![CI](https://img.shields.io/github/actions/workflow/status/Hiddentale/vulkan_rs/ci.yml?branch=main&logo=github&label=CI)](https://github.com/Hiddentale/vulkan_rs/actions)
[![docs.rs](https://img.shields.io/docsrs/vulkan-rs?logo=docs.rs&label=docs.rs)](https://docs.rs/vulkan-rs)
[![crates.io](https://img.shields.io/crates/v/vulkan-rs?logo=rust)](https://crates.io/crates/vulkan-rs)
[![License](https://img.shields.io/crates/l/vulkan-rs)](LICENSE-MIT)

Ergonomic, thin Vulkan bindings for Rust, generated from `vk.xml`.

All commands are inherent methods on `Entry`, `Instance`, and `Device`, no trait imports needed.
Supports `from_raw_parts` for OpenXR and middleware interop, complete command loading for
every enabled extension, and 100% documentation coverage enforced in CI.

## Getting Started

See the [examples](vulkan-rs/examples) directory for progressively complex Vulkan programs, from minimal setup to textured rendering. Run them with `cargo run --example hello_triangle_4`.

If you're new to Vulkan, start with the [Hello Triangle tutorial](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html) in the companion guide.

## Why vulkan-rs?

| | vulkan-rs | ash | vulkanalia |
|---|---|---|---|
| Command style | **Inherent methods** | Trait-based | Trait-based |
| `from_raw_parts` | **Yes, dedicated API** | Yes | No |
| Documentation | **100% coverage, spec links, examples** | Spec links only | Spec links + tutorial |
| Command loading | All enabled extensions | All enabled extensions | All enabled extensions |
| `no_std` (sys crate) | Yes | Yes | Yes |

**Key differences from ash:**

- **No trait imports.** `device.create_buffer(...)` works directly, no `use DeviceV1_0`.
- **`from_raw_parts` everywhere.** Construct `Instance`/`Device` from raw handles + `vkGetInstanceProcAddr`. Critical for OpenXR (`XR_KHR_vulkan_enable2`), middleware, and testing.
- **World-class documentation.** Every public item documented with spec links, usage notes, and examples. A companion [guide](https://hiddentale.github.io/vulkan_rs/) covers Vulkan concepts in depth.

## Crate Structure

```
vulkan-rs          Ergonomic wrapper (Entry, Instance, Device)
vulkan-rs-sys      Generated #[repr(C)] types from vk.xml (no_std)
vulkan-rs-codegen  Code generator (not published)
```

Users depend on `vulkan-rs` and access raw types via `vulkan_rs::vk::*`.

## Supported Platforms

| Platform | Library | Surface | CI Tested |
|----------|---------|---------|-----------|
| Windows | `vulkan-1.dll` | Win32 | Yes |
| Linux | `libvulkan.so.1` | X11, Wayland, XCB | Yes (+ lavapipe) |
| macOS | `libvulkan.1.dylib` | Metal | Yes |
| Android | `libvulkan.so` | Android | Build only |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `surface` | Yes | Window surface creation via `raw-window-handle`. Disable with `default-features = false` for headless use. |

## Examples

Progressive examples from minimal setup to textured rendering:

| Example | Lines | Concepts |
|---------|-------|----------|
| [hello_triangle_1](vulkan-rs/examples/hello_triangle_1.rs) | 81 | Entry, Instance |
| [hello_triangle_2](vulkan-rs/examples/hello_triangle_2.rs) | 248 | Surface, Swapchain |
| [hello_triangle_3](vulkan-rs/examples/hello_triangle_3.rs) | 413 | Pipeline, Command buffers |
| [hello_triangle_4](vulkan-rs/examples/hello_triangle_4.rs) | 588 | Complete triangle |
| [push_constants](vulkan-rs/examples/push_constants.rs) | 619 | Dynamic pipeline parameters |
| [double_buffering](vulkan-rs/examples/double_buffering.rs) | 620 | Triple-buffered rendering |
| [resize](vulkan-rs/examples/resize.rs) | 702 | Window resize handling |
| [textures](vulkan-rs/examples/textures.rs) | 930 | Image loading, sampling |

Run with `cargo run --example hello_triangle_4`.

## Guide

The companion [vulkan_rs Guide](https://hiddentale.github.io/vulkan_rs/) covers:

**Getting Started** - [Installation](https://hiddentale.github.io/vulkan_rs/getting-started/installation.html) | [Hello Triangle 1-4](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html)

**Concepts** - [Object Model](https://hiddentale.github.io/vulkan_rs/concepts/object-model.html) | [Memory](https://hiddentale.github.io/vulkan_rs/concepts/memory.html) | [Synchronization](https://hiddentale.github.io/vulkan_rs/concepts/synchronization.html) | [Render Passes](https://hiddentale.github.io/vulkan_rs/concepts/render-passes.html) | [Pipelines](https://hiddentale.github.io/vulkan_rs/concepts/pipelines.html) | [Descriptors](https://hiddentale.github.io/vulkan_rs/concepts/descriptors.html) | [Command Buffers](https://hiddentale.github.io/vulkan_rs/concepts/command-buffers.html) | [pNext Chains](https://hiddentale.github.io/vulkan_rs/concepts/pnext.html) | [Validation](https://hiddentale.github.io/vulkan_rs/concepts/validation.html)

**Architecture** - [Design Decisions](https://hiddentale.github.io/vulkan_rs/architecture/design.html) | [Error Handling](https://hiddentale.github.io/vulkan_rs/architecture/errors.html)

**How-To** - [Migrate from ash](https://hiddentale.github.io/vulkan_rs/how-to/migrate-from-ash.html) | [C-to-Rust Reference](https://hiddentale.github.io/vulkan_rs/how-to/c-to-rust.html) | [Push Constants](https://hiddentale.github.io/vulkan_rs/how-to/push-constants.html) | [Double Buffering](https://hiddentale.github.io/vulkan_rs/how-to/double-buffering.html) | [Resize](https://hiddentale.github.io/vulkan_rs/how-to/resize.html) | [Textures](https://hiddentale.github.io/vulkan_rs/how-to/textures.html)

## Testing

[Information about testing](docs/testing.md), including where tests of various kinds live and how to run them.

## MSRV Policy

The minimum supported Rust version is **1.85** (edition 2024).

MSRV bumps are treated as breaking changes and only happen in minor version increments (pre-1.0) or major version increments (post-1.0). Enforced in CI.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for documentation standards and the deprecation policy.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
