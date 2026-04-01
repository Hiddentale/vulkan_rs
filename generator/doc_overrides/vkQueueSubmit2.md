# Usage Notes

Vulkan 1.3 version of `queue_submit` that uses `SubmitInfo2` with
per-semaphore stage masks and 64-bit pipeline stage flags.

Key improvements over `queue_submit`:

- **Per-semaphore stage masks**: each wait semaphore has its own
  stage mask in `SemaphoreSubmitInfo`, instead of a parallel array.
  Clearer and less error-prone.
- **64-bit stages**: supports newer pipeline stages.
- **Timeline semaphores**: timeline values are embedded in
  `SemaphoreSubmitInfo` instead of requiring a separate pNext
  chain.

Prefer this over `queue_submit` when targeting Vulkan 1.3+. The
fence parameter works identically.
