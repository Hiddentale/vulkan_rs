# Usage Notes

Signals a timeline semaphore from the host (CPU), advancing its
counter to the specified value. The value must be greater than the
current counter value.

Use this to unblock GPU work that is waiting on the semaphore via
`queue_submit`. For example, a CPU-side data preparation step can
signal a timeline semaphore when data is ready, and the GPU waits on
it before processing.

Only valid for semaphores created with `SEMAPHORE_TYPE_TIMELINE`.

Timeline semaphores replace many use cases that previously required
fences, they can be waited on from both the CPU (`wait_semaphores`)
and the GPU (`queue_submit`).
