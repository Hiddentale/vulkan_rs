# Usage Notes

Samples multiple time domains simultaneously and returns
calibrated timestamps. This allows correlating GPU timestamps
(from `cmd_write_timestamp2`) with CPU time.

Each `CalibratedTimestampInfoKHR` specifies a time domain
(e.g., `DEVICE`, `CLOCK_MONOTONIC`, `CLOCK_MONOTONIC_RAW`,
`QUERY_PERFORMANCE_COUNTER`). All requested timestamps are
sampled as close together as possible.

The returned `max_deviation` (in nanoseconds) bounds how far
apart the samples could be, smaller is better. If deviation
is too large, retry the call.

Query available time domains first with
`get_physical_device_calibrateable_time_domains_khr`.
