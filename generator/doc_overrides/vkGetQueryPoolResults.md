# Usage Notes

Reads query results from a query pool into a host buffer. This is
the CPU-side retrieval path — for GPU-side copies into a device
buffer, use `cmd_copy_query_pool_results` instead.

**Key flags**:

- `QUERY_RESULT_64`: return 64-bit results. Always use this for
  timestamp and pipeline statistics queries to avoid overflow.
- `QUERY_RESULT_WAIT`: block until all requested queries are
  available. Without this flag, unavailable queries return
  `VK_NOT_READY` and their slots are left untouched.
- `QUERY_RESULT_WITH_AVAILABILITY`: append an availability value
  after each result (non-zero if available). Useful for polling
  without blocking.
- `QUERY_RESULT_PARTIAL`: return whatever data is available even
  for incomplete queries. Only meaningful for occlusion queries.

**Stride**: the `stride` parameter is the byte distance between
successive query results in your output buffer. It must be at least
large enough to hold the result plus the optional availability value.

Queries that have not been started or not yet completed return
`VK_NOT_READY` unless `QUERY_RESULT_WAIT` is set.
