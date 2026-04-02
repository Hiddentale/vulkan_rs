# Usage Notes

The input must be valid SPIR-V bytecode. The `code` slice is `&[u32]`
and must be aligned to 4 bytes, which Rust's `&[u32]` guarantees.

Use the `bytecode::read_spv` helper to load a `.spv` file from disk
with correct alignment.

Shader modules can be destroyed immediately after pipeline creation;
the driver copies what it needs during `create_graphics_pipelines` or
`create_compute_pipelines`. Destroying early keeps the handle count
low.
