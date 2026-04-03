# Changelog

All notable changes to vulkan-rust will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
