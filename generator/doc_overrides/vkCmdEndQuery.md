# Usage Notes

Ends an active query at the specified index. The query results
become available for retrieval via `get_query_pool_results` or
`cmd_copy_query_pool_results` once the command buffer has completed
execution.

Must be paired with a preceding `cmd_begin_query` on the same
query index. Beginning a query without ending it, or ending one
that was not begun, is an error.
