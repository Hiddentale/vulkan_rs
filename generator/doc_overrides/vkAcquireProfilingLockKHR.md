# Usage Notes

Acquires the device profiling lock, which must be held while
submitting command buffers that contain performance queries.
Only one thread can hold the lock at a time.

The `AcquireProfilingLockInfoKHR` specifies a timeout in
nanoseconds. Returns `TIMEOUT` if the lock cannot be acquired
within that period.

Release with `release_profiling_lock_khr` when profiling
submission is complete.

Requires `VK_KHR_performance_query`.
