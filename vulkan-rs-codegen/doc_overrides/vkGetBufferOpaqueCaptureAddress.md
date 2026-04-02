# Usage Notes

Returns an opaque capture address for a buffer that was created with
`BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY`. This address is used
to recreate the buffer at the same virtual address in a replay
session (e.g. for GPU crash dump replay or deterministic replay
tools).

Most applications do not need this, it is primarily for debugging
and profiling tools. Use `get_buffer_device_address` for runtime
buffer address access.
