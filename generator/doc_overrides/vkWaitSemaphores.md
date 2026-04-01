# Usage Notes

Blocks the calling thread until one or all of the specified timeline
semaphores reach their target values, or until the timeout expires.

**`SemaphoreWaitInfo` flags**:

- `SEMAPHORE_WAIT_ANY`: return when *any* semaphore reaches its
  target. Without this flag, waits for *all* semaphores.

**Timeout**: in nanoseconds. `u64::MAX` for indefinite. Zero for a
non-blocking poll.

Timeline semaphore waits are the CPU-side counterpart to
`queue_submit` timeline waits. They replace many fence-based
synchronisation patterns with a single, more flexible primitive.

```text
// Wait for frame N to complete on the GPU
let info = SemaphoreWaitInfo::builder()
    .semaphores(&[timeline_sem])
    .values(&[frame_number]);
wait_semaphores(&info, u64::MAX);
```

Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.
