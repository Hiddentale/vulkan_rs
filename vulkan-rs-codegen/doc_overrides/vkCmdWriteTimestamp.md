# Usage Notes

Writes the current GPU timestamp into a query pool slot when the
specified pipeline stage completes. Use two timestamps to measure
elapsed GPU time:

```text
cmd_write_timestamp(PIPELINE_STAGE_TOP_OF_PIPE, pool, 0);
// ... commands to measure ...
cmd_write_timestamp(PIPELINE_STAGE_BOTTOM_OF_PIPE, pool, 1);
```

After the command buffer completes, read the values with
`get_query_pool_results` (with `QUERY_RESULT_64`) and compute:

```text
elapsed_ns = (timestamp[1] - timestamp[0]) * timestamp_period
```

`timestamp_period` is in nanoseconds per tick, available from
`physical_device_limits`.

Not all queue families support timestamps, check
`timestamp_valid_bits` in the queue family properties. A value of 0
means timestamps are not supported on that queue.

For Vulkan 1.3+, prefer `cmd_write_timestamp2`.
