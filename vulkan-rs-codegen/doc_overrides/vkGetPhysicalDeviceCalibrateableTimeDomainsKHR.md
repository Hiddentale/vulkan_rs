# Usage Notes

Enumerates the time domains that can be used with
`get_calibrated_timestamps_khr` on this physical device.

Common time domains include:

- `DEVICE`: GPU timestamp counter (same as `cmd_write_timestamp2`).
- `CLOCK_MONOTONIC` / `CLOCK_MONOTONIC_RAW`: Linux monotonic
  clocks.
- `QUERY_PERFORMANCE_COUNTER`: Windows high-resolution timer.

The device time domain is always available. Host time domains
depend on the platform.
