# Usage Notes

Resets a range of queries in a pool from the GPU command stream.
Queries must be reset before they can be used in `cmd_begin_query`
or `cmd_write_timestamp`.

This is the pre-1.2 way to reset queries. For Vulkan 1.2+,
`reset_query_pool` (host-side) is often more convenient and avoids
adding the reset to the command buffer.

Must be recorded outside a render pass.
