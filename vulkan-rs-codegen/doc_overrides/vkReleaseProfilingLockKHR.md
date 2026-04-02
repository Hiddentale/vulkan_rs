# Usage Notes

Releases the device profiling lock previously acquired with
`acquire_profiling_lock_khr`. Must be called after all command
buffers containing performance queries have been submitted.

Requires `VK_KHR_performance_query`.
