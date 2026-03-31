# Contributing to vulkan_rs

## Documentation Template

All new public items in `vk-engine` must follow this section order:

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
/// - [`vk::enums::Result::ERROR_OUT_OF_HOST_MEMORY`] — Host allocation failed.
///
/// # Safety
///
/// The caller must ensure all shader modules in `info.stages` were
/// created from the same [`Device`] as `self`.
///
/// # Examples
///
/// ```no_run
/// # use vk_engine::*;
/// // example code here
/// ```
```

Omit sections that don't apply (e.g., no `# Safety` on safe functions).

## Generated vs Hand-Written Code

- `vk-sys/src/` — fully generated. Do not edit by hand. Run `cargo run -p generator`.
- `vk-engine/src/generated/` — fully generated. Same as above.
- `vk-engine/src/*.rs` (non-generated) — hand-written, highest doc quality bar.
- `generator/doc_overrides/` — hand-written doc additions appended to generated wrappers.

## Before Submitting

- [ ] All new public items have doc comments
- [ ] `# Safety` section on all `unsafe` functions
- [ ] `# Errors` section on all `Result`-returning functions
- [ ] Examples compile: `cargo test --doc`
- [ ] No broken doc links: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
- [ ] All tests pass: `cargo test --workspace`
- [ ] Generated output is up to date: `cargo run -p generator` then check `git diff`
