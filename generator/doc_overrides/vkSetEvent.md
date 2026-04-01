# Usage Notes

Signals an event from the host (CPU). After this call,
`get_event_status` returns `VK_EVENT_SET`.

Host-signaled events are primarily useful for host–host
synchronisation or as a manual control mechanism. For GPU–GPU
synchronisation, prefer `cmd_set_event` recorded in a command
buffer.
