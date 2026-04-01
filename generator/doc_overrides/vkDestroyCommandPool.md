# Usage Notes

Destroys a command pool and implicitly frees all command buffers
allocated from it. You do not need to free individual command buffers
before destroying the pool.

All command buffers allocated from this pool must have completed
execution before the pool is destroyed. Call `device_wait_idle` or
wait on the relevant fences first.
