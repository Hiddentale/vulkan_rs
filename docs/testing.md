# Testing in vulkan-rs

## Run All Tests

```bash
# Unit tests + integration tests (no Vulkan runtime needed)
cargo test --workspace

# Runtime tests (requires Vulkan driver or lavapipe)
cargo test --workspace -- --ignored --test-threads=1
```

## Test Categories

### Unit Tests (mock-based)

- Located throughout `vulkan-rs/src/*.rs` in `#[cfg(test)]` modules.
- Run with `cargo test -p vulkan-rs --lib`
- No Vulkan runtime required.

These use mock loaders (`NullLoader`, `FakeEntryLoader`, `RichEntryLoader`,
`FailingEntryLoader`) that return fake function pointers via
`extern "system" fn` stubs. This lets us test the full Entry -> Instance ->
Device creation pipeline, error propagation, Arc lifecycle, and command
loading without touching a real GPU.

Each module has both happy-path and error-path mocks. The mocks are
intentionally verbose (explicit `transmute` per function pointer) to make
the test's assumptions visible.

### Integration Tests

- Located in `vulkan-rs/tests/`.
- Run with `cargo test -p vulkan-rs --tests`

| File | What it tests |
|------|---------------|
| `bytecode_integration.rs` | SPIR-V alignment edge cases |
| `entry_integration.rs` | Entry creation with mock loaders |
| `mock_wrapper_tests.rs` | Generated wrapper methods via mocks |
| `generated_wrappers_integration.rs` | Runtime wrapper behavior |
| `instance_device_integration.rs` | Full pipeline with real Vulkan (`#[ignore]`) |

### Runtime Tests (requires Vulkan)

- Marked with `#[ignore]` throughout `vulkan-rs/src/*.rs` and `vulkan-rs/tests/`.
- Run with `cargo test --workspace -- --ignored --test-threads=1`
- CI uses lavapipe (Mesa software Vulkan driver) on Ubuntu.

These tests create real Vulkan objects and exercise the actual driver.
They must run single-threaded (`--test-threads=1`) because NVIDIA's
implicit layers are not thread-safe during concurrent `vkCreateInstance`
calls. A shared `VK_TEST_MUTEX` guards against this in unit tests.

To run locally, you need a Vulkan driver installed. On Linux without a
GPU, install lavapipe:

```bash
sudo apt-get install mesa-vulkan-drivers
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name '*lvp*' -o -name '*lavapipe*' 2>/dev/null | head -1)
cargo test --workspace -- --ignored --test-threads=1
```

### Doc Tests

- Run with `cargo test --doc --all-features`
- Most use `no_run` (compile-only) since they need a Vulkan runtime.
- The `test_helpers` module provides `create_test_entry()`,
  `create_test_instance()`, and `create_test_device()` to reduce
  boilerplate in doc examples.

### Fuzz Tests

- Located in `vulkan-rs/fuzz/fuzz_targets/`.
- Run with `cd vulkan-rs && cargo +nightly fuzz run <target>`
- CI runs each target for 60 seconds per PR.

| Target | What it fuzzes |
|--------|---------------|
| `fuzz_cast_to_u32` | SPIR-V byte alignment validation (property-based) |
| `fuzz_entry_loader` | Loader trait with randomized symbol resolution modes |
| `fuzz_push_next` | Builder pNext chain ordering and linked-list integrity |

### Layout Cross-Validation

- C source: `vulkan-rs-sys/tests/c_layout_check.c`
- Rust source: `vulkan-rs-sys/src/bin/rust_layout_check.rs`
- Run in CI only (requires matching Vulkan headers).

Compiles a C program and a Rust program that both print struct sizes and
field offsets, then diffs the output. Catches ABI mismatches between the
generated Rust types and the C Vulkan headers. The CI job clones
Vulkan-Headers at the exact version tag matching our vendored `vk.xml`.

### Generator Determinism

- Run with `cargo run -p vulkan-rs-codegen && git diff --quiet`
- CI fails if the generated output has changed.

Re-runs the code generator and checks that the committed generated files
are up to date. Prevents stale generated code from being merged.

### vulkan-rs-sys Tests

- Located in `vulkan-rs-sys/tests/`.
- Run with `cargo test -p vulkan-rs-sys`

| File | What it tests |
|------|---------------|
| `handles.rs` | Handle trait: `null()`, `is_null()`, `from_raw()`, `as_raw()` round-trips |
| `layout.rs` | Struct size and alignment for critical types |

## Writing New Tests

- **Mock-based unit tests** are preferred for new wrapper logic. Create a
  mock `vkGetInstanceProcAddr` or `vkGetDeviceProcAddr` that returns fake
  function pointers for the commands under test.
- **`#[ignore]` tests** for anything that needs a real Vulkan driver.
  Always acquire `VK_TEST_MUTEX` first.
- **One behavior per test function.** Name it
  `test_<method>_<scenario>_<expected_outcome>`.
- **Fuzz targets** for any function that processes untrusted input
  (byte slices, raw pointers, user-provided data).
