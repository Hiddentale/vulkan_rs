# Changelog

All notable changes to vulkan-rust will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-04-03

### Added

- Builders for all plain (non-sType) structs, not just extensible ones (~200 new builders)
- `all()` method on bitmask types, combining every defined flag
- `Display` and `Error` impls on generated enum types (`vk::Result` now works with `?` and anyhow)
- `DeviceSize` and `DeviceAddress` type aliases
- `ClearValue::color_f32()`, `ClearValue::depth_stencil()`, and other typed constructors for union types
- `PartialOrd`/`Ord` derives on bitmask types

### Changed

- **Breaking:** All `vk::` submodules (`handles`, `structs`, `enums`, `bitmasks`, `constants`, `builders`) are now private with glob re-exports at the crate root. Use `vk::Buffer` instead of `vk::handles::Buffer`.
- **Breaking:** Builder setter methods for pointer fields drop the `p_` prefix (`p_vertex_input_state()` becomes `vertex_input_state()`, `p_application_info()` becomes `application_info()`)
- **Breaking:** `get_physical_device_surface_support_khr` returns `VkResult<bool>` instead of `VkResult<u32>`
- **Breaking:** `cmd_push_constants` and other void-array commands accept `&[u8]` instead of `&[c_void]`
- **Breaking:** Optional `const char*` params (e.g. `enumerate_device_extension_properties`) accept `Option<&CStr>` instead of `*const c_char`
- `ShaderModuleCreateInfo` builder has a single `.code(&[u32])` setter instead of separate `.code_size()` and `.p_code()`
- Multiple input arrays sharing a count param (e.g. `cmd_bind_vertex_buffers`) are all emitted as slices
- Count field setters are preserved when the paired pointer is optional (e.g. `descriptor_count` on `DescriptorSetLayoutBinding`)

### Fixed

- PFN callback types (`PFN_vkDebugUtilsMessengerCallbackEXT`, etc.) are now fully typed instead of type-erased `fn()`
- Parser prefers `altlen` over `len` from vk.xml, giving parseable C expressions instead of LaTeX
- `#[doc(inline)]` on all re-exports so docs.rs renders type pages under `vk::`
- README links use absolute GitHub URLs for crates.io compatibility
- Guide dependency version corrected from `"0.1"` to `"0.10"`

## [0.9.0] - 2026-04-03

Initial public release.

### Added

- Code generator from `vk.xml` (all core versions, all extensions, all platforms)
- `vulkan-rust-sys`: generated `#[repr(C)]` types, handles, enums, bitmasks, builders, command tables (`no_std`)
- `vulkan-rust`: ergonomic wrapper with `Entry`, `Instance`, `Device`
- `from_raw_parts` on `Instance` and `Device` for OpenXR/middleware interop
- Inherent methods for all ~722 Vulkan commands (no trait imports)
- Builder pattern with `push_next` for type-safe pNext chains
- Platform surface creation (Win32, X11, XCB, Wayland, Metal, Android)
- SPIR-V bytecode alignment helper (`cast_to_u32`)
- C-to-Rust struct layout cross-validation in CI
- 9 progressive examples (hello triangle through textures)
- Companion mdBook guide with 22 chapters
- 100% documentation coverage enforced in CI
