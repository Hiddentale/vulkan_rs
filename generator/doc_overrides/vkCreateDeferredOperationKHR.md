# Usage Notes

Creates a deferred operation handle. Deferred operations allow
expensive host-side work (such as ray tracing pipeline compilation)
to be split across multiple CPU threads.

The typical workflow:

1. Create a deferred operation with this command.
2. Pass the handle to a deferrable command (e.g.,
   `create_ray_tracing_pipelines_khr`). If deferred, it returns
   `OPERATION_DEFERRED_KHR`.
3. Query `get_deferred_operation_max_concurrency_khr` to learn
   how many threads can contribute.
4. Call `deferred_operation_join_khr` from each worker thread.
5. Once all joins return `SUCCESS`, retrieve the result with
   `get_deferred_operation_result_khr`.
6. Destroy the handle with `destroy_deferred_operation_khr`.

The handle itself is lightweight, it is just a token for tracking
the deferred work.
