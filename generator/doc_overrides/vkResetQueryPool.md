# Usage Notes

Resets a range of queries in a pool from the host (CPU). This is the
Vulkan 1.2 host-side alternative to `cmd_reset_query_pool`, which
resets queries from a command buffer.

Host-side reset is simpler, call it directly without recording a
command buffer. Requires the `host_query_reset` feature (core in
Vulkan 1.2).

Queries must be reset before use. Resetting a query that is in use
by a pending command buffer is an error.
