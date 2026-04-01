# Usage Notes

Waits for one or more events to be signaled and then inserts memory
and execution dependencies. This is the "wait" half of the
signal/wait pattern started by `cmd_set_event`.

The wait blocks execution at the specified destination pipeline
stages until all events are signaled. Memory barriers provided in
this call make the specified source writes visible to the
destination stages.

**Split barriers**: the main advantage over `cmd_pipeline_barrier`
is that you can interleave unrelated commands between the signal and
wait, giving the GPU more opportunity for parallel execution.

Events must not be waited on across different queues. For
cross-queue synchronisation, use semaphores.

For Vulkan 1.3+, prefer `cmd_wait_events2`.
