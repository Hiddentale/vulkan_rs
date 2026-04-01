# Usage Notes

Returns the maximum number of threads that can usefully join this
deferred operation. Spawning more threads than this value wastes
resources, additional joins will return `THREAD_IDLE_KHR`.

The returned value may decrease over time as work completes, so
query it just before spawning worker threads.

A return value of zero means the operation is already complete
or requires no additional threads.
