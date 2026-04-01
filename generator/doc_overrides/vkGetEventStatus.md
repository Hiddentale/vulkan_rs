# Usage Notes

Returns whether an event is currently signaled or unsignaled.
Returns `VK_EVENT_SET` if signaled, `VK_EVENT_RESET` if not.

This is a non-blocking host-side query. Use it to poll for
GPU-signaled events when you need to know the result without
blocking. For blocking synchronisation, use fences instead.
