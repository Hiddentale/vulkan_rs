# Usage Notes

Signals an event from the GPU at a specific pipeline stage. A later
`cmd_wait_events` call can wait for this signal to synchronise work
within the same queue.

Events provide finer-grained synchronisation than pipeline barriers
when you want to split a dependency into a "signal" point and a
"wait" point separated by other commands. This lets the GPU execute
interleaving work between the signal and wait.

The `stage_mask` specifies at which pipeline stage the event is
signaled. The event becomes signaled once all commands prior to this
call have completed that stage.

Events must only be used within a single queue. For cross-queue
synchronisation, use semaphores.

For Vulkan 1.3+, prefer `cmd_set_event2` which supports more
precise stage and access masks.
