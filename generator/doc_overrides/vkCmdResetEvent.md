# Usage Notes

Resets an event to the unsignaled state from the GPU at a specific
pipeline stage. The event can then be signaled again by a
subsequent `cmd_set_event`.

Must not be called while a `cmd_wait_events` that waits on this
event is between its wait and the completion of the dependent work.

For Vulkan 1.3+, prefer `cmd_reset_event2`.
