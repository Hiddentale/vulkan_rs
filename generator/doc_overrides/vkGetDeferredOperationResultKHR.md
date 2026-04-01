# Usage Notes

Returns the result of a completed deferred operation. The returned
`VkResult` is the same value that the original deferrable command
would have returned if it had executed synchronously.

Only call this after the operation has fully completed (all joins
returned `SUCCESS` or `THREAD_DONE_KHR`). Calling on an
in-progress operation returns `NOT_READY`.

For example, if `create_ray_tracing_pipelines_khr` was deferred,
this returns whether pipeline creation succeeded or failed.
