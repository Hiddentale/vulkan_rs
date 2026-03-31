# Usage Notes

Destroys a semaphore. The semaphore must not be referenced by any
pending queue submission — either as a wait or signal semaphore.

Wait for all submissions that use this semaphore to complete (via
fences or `device_wait_idle`) before destroying it.
