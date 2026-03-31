# Usage Notes

Blocks the calling thread until all commands submitted to this queue
have completed. Equivalent to submitting a fence and immediately
waiting on it, but simpler.

Use this for quick synchronization in non-performance-critical paths
(e.g. during teardown or after a one-shot transfer). In a render
loop, prefer fences or timeline semaphores for finer-grained
control — `queue_wait_idle` stalls the CPU and prevents overlap
between CPU and GPU work.

The queue must be externally synchronized: do not call this while
another thread is submitting to the same queue.
