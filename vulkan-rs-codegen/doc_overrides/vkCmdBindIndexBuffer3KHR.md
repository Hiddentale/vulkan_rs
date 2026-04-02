# Usage Notes

Device-address variant of `cmd_bind_index_buffer`. Binds an index
buffer for subsequent indexed draw commands using a device address
instead of a buffer handle.

The `BindIndexBuffer3InfoKHR` struct specifies the device address,
size, and index type (`UINT16`, `UINT32`, or `UINT8` if enabled).

Supersedes `cmd_bind_index_buffer` and `cmd_bind_index_buffer2`
when using `VK_KHR_device_address_commands`.
