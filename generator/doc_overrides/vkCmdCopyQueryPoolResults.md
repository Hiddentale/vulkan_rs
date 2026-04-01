# Usage Notes

Copies query results directly into a GPU buffer. This is the
GPU-side counterpart to `get_query_pool_results` and avoids a CPU
round-trip when the results are consumed by subsequent GPU work
(e.g. conditional rendering or indirect dispatch).

The same flags apply as for `get_query_pool_results`:
`QUERY_RESULT_64`, `QUERY_RESULT_WAIT`,
`QUERY_RESULT_WITH_AVAILABILITY`, and `QUERY_RESULT_PARTIAL`.

The destination buffer must have `BUFFER_USAGE_TRANSFER_DST`. The
stride must be large enough to hold the result (and availability
value, if requested).

Must be recorded outside a render pass.
