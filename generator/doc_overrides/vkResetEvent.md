# Usage Notes

Resets an event to the unsignaled state from the host (CPU). The
event must not be waited on by any pending `cmd_wait_events` call.

After resetting, the event can be signaled again by `set_event` or
`cmd_set_event`.
