# Usage Notes

Submits anti-lag timing data to reduce input-to-display latency.
Called once per frame with presentation timing hints so the driver
can pace GPU work to minimise latency.

Requires `VK_AMD_anti_lag`.
