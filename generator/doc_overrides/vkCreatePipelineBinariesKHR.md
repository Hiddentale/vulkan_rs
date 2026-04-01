# Usage Notes

Creates pipeline binary objects from either a pipeline create info
or previously serialized binary data. Pipeline binaries capture
compiled shader code in a device-specific format, enabling fast
pipeline recreation without recompilation.

Two creation paths via `PipelineBinaryCreateInfoKHR`:

- **From pipeline create info + key**: compiles shaders and
  produces binaries. Use `get_pipeline_key_khr` to obtain the
  key first.
- **From serialized data**: restores binaries saved with
  `get_pipeline_binary_data_khr` from a prior run. This skips
  compilation entirely.

The output is written to `PipelineBinaryHandlesInfoKHR`. Call
once with a null `pipelines` pointer to query the count, then
again with an allocated array.

Pipeline binaries are more portable than pipeline caches, they
can be validated, versioned, and stored in application-managed
files rather than opaque blobs.
