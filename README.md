# vulkan_rs
Minimal Vulkan 1.2 bindings for Rust.

Current progress:

- Generator from vk.xml with vk-parse + quote
- Two-crate split with directed dependency
- from_raw_parts interop
- Inherent methods (not traits)
- Complete command loading (all ~700 PFN typedefs)
- Builder Deref pattern with push_next
- #![no_std] for vk-sys
- CI with fmt/clippy/build/test

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
