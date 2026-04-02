# Usage Notes

Sets a latency marker at a specific point in the frame lifecycle
(simulation start, render start, present, etc.). The markers are
later retrieved with `get_latency_timings_nv` to measure per-
stage latency.

Requires `VK_NV_low_latency2`.
