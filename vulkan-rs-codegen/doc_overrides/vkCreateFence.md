# Usage Notes

Fences are the primary CPU–GPU synchronization primitive. The CPU
blocks on `wait_for_fences` until the GPU signals the fence.

**Initial state**: create with `FENCE_CREATE_SIGNALED` when the
fence is used in a frame loop that waits before the first submit.
Without this flag the first `wait_for_fences` would block forever.

**Typical frame loop pattern**:

1. `wait_for_fences`, block until the previous frame's GPU work
   completes.
2. `reset_fences`, reset back to unsignaled.
3. Record and submit commands, passing the fence to `queue_submit`.

A fence can only be associated with one submission at a time.
Submitting with a fence that is already pending is an error.

For GPU–GPU synchronization (between queue submissions) use
semaphores instead. Fences are strictly for CPU-visible signalling.
