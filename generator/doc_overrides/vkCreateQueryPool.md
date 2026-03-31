# Usage Notes

Creates a pool of query slots. Queries let you measure GPU
performance and gather statistics without stalling the pipeline.

**Query types**:

- `OCCLUSION`: counts how many samples pass the depth test. Useful
  for visibility culling — render a bounding box, check the count.
- `PIPELINE_STATISTICS`: counts shader invocations, primitives,
  clipping, etc. Must be enabled via
  `pipeline_statistics_query` device feature.
- `TIMESTAMP`: records a GPU timestamp. Use two timestamps and the
  `timestamp_period` device property to measure elapsed time.

Queries must be reset before use with `cmd_reset_query_pool` (or
`reset_query_pool` on Vulkan 1.2+). Results are retrieved with
`get_query_pool_results` or copied into a buffer with
`cmd_copy_query_pool_results`.
