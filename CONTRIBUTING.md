# Contributing to vulkan_rs

## Documentation Template

All new public items in `vulkan-rs` must follow this section order:

```rust
/// One-line summary in imperative mood.
///
/// Extended description: what this does, when to use it, relationship
/// to other methods. Links to [`RelatedType`] and
/// [`alternative_method`](Self::alternative_method).
///
/// # Parameters
///
/// - `info`: Pipeline configuration. See [`GraphicsPipelineCreateInfo`].
/// - `cache`: Optional [`PipelineCache`] to accelerate creation.
///
/// # Returns
///
/// A fully initialized [`GraphicsPipeline`] ready for command recording.
///
/// # Errors
///
/// - [`vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY`], Host allocation failed.
///
/// # Safety
///
/// The caller must ensure all shader modules in `info.stages` were
/// created from the same [`Device`] as `self`.
///
/// # Examples
///
/// ```no_run
/// # use vulkan_rs::*;
/// // example code here
/// ```
```

Omit sections that don't apply (e.g., no `# Safety` on safe functions).

## Generated vs Hand-Written Code

- `vulkan-rs-sys/src/`, fully generated. Do not edit by hand. Run `cargo run -p vulkan-rs-codegen`.
- `vulkan-rs/src/generated/`, fully generated. Same as above.
- `vulkan-rs/src/*.rs` (non-generated), hand-written, highest doc quality bar.
- `vulkan-rs-codegen/doc_overrides/`, hand-written doc additions appended to generated wrappers.

## Deprecation Policy

Public API items must never be removed without a deprecation cycle:

1. **Deprecate** in a minor release using the built-in attribute:
   ```rust
   #[deprecated(since = "0.3.0", note = "renamed to `create_instance_handle`")]
   pub unsafe fn create_instance_raw(...) -> VkResult<vk::handles::Instance> {
       self.create_instance_handle(...)
   }
   ```
2. **Keep** the deprecated item compiling and forwarding to the replacement for the duration of that minor version.
3. **Remove** the deprecated item in the next breaking version (the next `0.x.0` pre-1.0, or the next major version post-1.0).

This applies to all public functions, types, traits, re-exports, and feature flags. Internal (`pub(crate)`) items can be changed freely.

## Before Submitting

- [ ] All new public items have doc comments
- [ ] `# Safety` section on all `unsafe` functions
- [ ] `# Errors` section on all `Result`-returning functions
- [ ] Examples compile: `cargo test --doc`
- [ ] No broken doc links: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- [ ] All tests pass: `cargo test --workspace`
- [ ] Generated output is up to date: `cargo run -p vulkan-rs-codegen` then check `git diff`
