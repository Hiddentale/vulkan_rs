# Contributing to vulkan_rust

## Core Principle

Maintainability is the #1 concern for any new code contributed to this project. Every decision, from naming to structure to abstraction, should prioritize making the code easy to read, understand, and change in the future.

---

## Coding Standards

### Architecture

- **High cohesion, low coupling.** Each module has one clear responsibility. Functions and modules that change together live together; functions and modules that do not change together are separated.
- **Directed dependency hierarchy.** Dependencies flow one way: no circular imports, no coupling between sibling modules except through a shared interface.
- **Few top-level components.** Minimise the number of modules at each layer. A flat, wide structure is harder to navigate than a shallow hierarchy with clear boundaries.
- **Module size limit: 400 lines of production code.** Test code (`#[cfg(test)]` blocks, `tests/` files) does not count toward this limit. If a module's non-test code exceeds 400 lines, it is doing too much. Extract cohesive sub-functionality into a dedicated module.
- **No premature generalisation.** Solve the problem in front of you. Do not add abstractions, traits, or configuration knobs for requirements that do not yet exist.

### Functions

- **One level of abstraction per function.** A function that does high-level orchestration delegates all detail to sub-functions. A function that does low-level computation does not also orchestrate.
- **Extract cohesive sub-functionality into named sub-functions.** If a block of code inside a function deserves a comment to explain what it does, it deserves a name and a function boundary instead.
- **Short functions.** A function that does not fit on one screen is too long.
- **No side effects unless the function name says so.** `compute_*` and `get_*` are pure. `store_*`, `update_*`, and `create_*` have side effects.

### Naming

- **Names must be precise and short.** A long identifier is a sign the concept is not well understood.
- **No abbreviations** unless the abbreviation is universally understood in this domain (e.g. `vk`, `cmd`, `gpu`).
- **No redundant context in names.** If the module is `pipeline`, the function is `create`, not `create_pipeline`.

### Constants

- **No magic constants.** Every numeric or string literal that encodes a domain decision belongs in a named constant at the top of the module or in a dedicated `constants` module.
- **Name constants for what they mean, not what they are.** `MIN_API_VERSION: u32 = 1`, not `ONE`.

### Comments

- **No bad comments.** Do not state what the code does; state why it does it if the reason is not obvious.
- **No commented-out code.** Dead code goes in version control history, not in the file.
- **No dead code.** Unused functions, unused imports, unreachable branches, delete them.

### Error Handling

- **No swallowed errors.** Catching a `Result` and discarding the error (or only logging) is only acceptable if you explicitly document why silence is correct.
- **Let errors propagate** to the boundary where they can be meaningfully handled. Use `?` to propagate, do not match and re-wrap without adding context.
- **Fail loudly at invalid inputs.** Use `assert!`, `debug_assert!`, or return a descriptive error at function entry when preconditions are violated.

### Testing

#### Philosophy

- **Tests are first-class code.** They live in `#[cfg(test)]` modules or `tests/`, are subject to the same quality standards as production code, and are committed with the feature they test.
- **Testing shows presence of defects, not absence.** A passing test suite is not a proof of correctness. Design tests to find bugs, not to confirm the code runs.
- **Exhaustive testing is impossible.** Apply risk-based prioritisation: identify the modules most likely to fail and concentrate effort there. 80% of defects cluster in 20% of the code.
- **Test early, test often.** Write tests before or alongside the code they cover, not after.

#### What to test

- **All non-trivial logic must have an automated test.** If a function makes a decision, branches on data, or computes a non-obvious result, it is tested.
- **Trivial code (field access, pass-through delegation) does not need a dedicated test.**
- **Prioritise high-risk modules.** Code generation, Vulkan object lifecycle, and API surface construction are high risk. Re-exports and type aliases are lower risk.
- **Representative sample over exhaustive coverage.** Choose test cases that cover:
  - The normal / expected path
  - Boundary conditions (empty input, single element, maximum values)
  - Known failure modes (null handles, invalid enum variants, missing extensions)

#### How to write tests

- **One behaviour per test function.** A test that checks multiple independent behaviours hides which one failed.
- **Test names describe the scenario and expected outcome.** `test_create_instance_returns_error_when_layer_missing` not `test_instance_1`.
- **No logic in tests.** Loops, conditionals, and complex setup inside a test make it untrustworthy.
- **Shared setup in helper functions.** If three tests need the same fixture, put it in a helper function, not copy-pasted in each test body.
- **Do not mock unless crossing a real boundary** (Vulkan driver, filesystem, network). Do not mock internal helper functions.

---

## Documentation Template

All new public items in `vulkan-rust` must follow this section order:

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
/// - [`vk::Result::ERROR_OUT_OF_HOST_MEMORY`], Host allocation failed.
///
/// # Safety
///
/// The caller must ensure all shader modules in `info.stages` were
/// created from the same [`Device`] as `self`.
///
/// # Examples
///
/// ```no_run
/// # use vulkan_rust::*;
/// // example code here
/// ```
```

Omit sections that don't apply (e.g., no `# Safety` on safe functions).

## Generated vs Hand-Written Code

- `vulkan-rust-sys/src/`, fully generated. Do not edit by hand. Run `cargo run -p vulkan-rust-codegen`.
- `vulkan-rust/src/generated/`, fully generated. Same as above.
- `vulkan-rust/src/*.rs` (non-generated), hand-written, highest doc quality bar.
- `vulkan-rust-codegen/doc_overrides/`, hand-written doc additions appended to generated wrappers.

## Deprecation Policy

Public API items must never be removed without a deprecation cycle:

1. **Deprecate** in a minor release using the built-in attribute:
   ```rust
   #[deprecated(since = "0.3.0", note = "renamed to `create_instance_handle`")]
   pub unsafe fn create_instance_raw(...) -> VkResult<vk::Instance> {
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
- [ ] Generated output is up to date: `cargo run -p vulkan-rust-codegen` then check `git diff`
