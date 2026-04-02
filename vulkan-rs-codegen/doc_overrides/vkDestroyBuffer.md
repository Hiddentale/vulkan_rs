# Usage Notes

Destroys a buffer object. The buffer must not be in use by any
pending GPU work, wait on the relevant fences or call
`device_wait_idle` before destroying.

Destroying a buffer does **not** free its backing memory. Call
`free_memory` separately (or let your sub-allocator reclaim the
region).

Destroy order: destroy the buffer first, then free the memory. Not
the reverse, freeing memory while a buffer is still bound to it is
undefined behaviour.
