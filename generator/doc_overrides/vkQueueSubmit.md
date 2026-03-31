# Usage Notes

`queue_submit` is the primary way to send recorded command buffers
to the GPU. Each `SubmitInfo` specifies:

- **Wait semaphores + stage masks**: the submission waits on these
  semaphores at the given pipeline stages before executing.
- **Command buffers**: executed in array order within the submission.
- **Signal semaphores**: signalled when all command buffers complete.

**Fence**: pass a fence to know when the *entire batch* of submissions
completes on the CPU side. Passing `Fence::null()` means there is no
CPU-visible signal — you must use semaphores or `queue_wait_idle`
instead.

Minimize `queue_submit` calls. Each call has driver overhead; batching
multiple `SubmitInfo` entries into one call is cheaper than separate
calls.

**Thread safety**: a `Queue` must be externally synchronized. If
multiple threads submit to the same queue, you need a mutex.
