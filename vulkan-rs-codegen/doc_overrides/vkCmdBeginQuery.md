# Usage Notes

Begins a query at the specified index in a query pool. All
rendering or compute commands recorded between `cmd_begin_query` and
`cmd_end_query` are measured by the query.

**Flags**:

- `QUERY_CONTROL_PRECISE`: for occlusion queries, return an exact
  sample count instead of a boolean. More expensive on some
  hardware. Requires the `occlusion_query_precise` device feature.

The query slot must have been reset with `cmd_reset_query_pool` (or
`reset_query_pool` on Vulkan 1.2+) before beginning.

Pipeline statistics queries must be begun and ended outside a render
pass. Occlusion queries can span draw calls within a render pass.
