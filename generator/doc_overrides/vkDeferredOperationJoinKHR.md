# Usage Notes

Joins the calling thread to a deferred operation, contributing
CPU time to its completion. Multiple threads can join the same
operation concurrently.

Return values:

- `SUCCESS`, the operation completed. The calling thread was
  the last one needed.
- `THREAD_DONE_KHR`, this thread's contribution is finished,
  but the operation may still be in progress on other threads.
- `THREAD_IDLE_KHR`, the operation has enough threads; this
  one was not needed. Retry later or move on.

Call this in a loop per thread until it returns `SUCCESS` or
`THREAD_DONE_KHR`. After all threads finish, check the final
result with `get_deferred_operation_result_khr`.

The number of useful threads is bounded by
`get_deferred_operation_max_concurrency_khr`.
