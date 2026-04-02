# Usage Notes

Serializes a pipeline binary into a byte buffer for offline
storage. The data can be saved to disk and later passed to
`create_pipeline_binaries_khr` to skip shader compilation on
subsequent application launches.

Uses the two-call pattern: call with a null `p_pipeline_binary_data`
to query the required `data_size`, allocate a buffer, then call
again to fill it.

The output also includes a `PipelineBinaryKeyKHR` that identifies
the binary. Store the key alongside the data, it is required
when recreating the binary.

Serialized data is device-specific and may become invalid after
driver updates. Applications should handle creation failure
gracefully by falling back to full recompilation.
