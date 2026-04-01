# Usage Notes

Exports a device memory allocation as a POSIX file descriptor.
The fd can be sent to another process (via Unix domain sockets)
or another Vulkan device to share GPU memory.

`MemoryGetFdInfoKHR` specifies the `DeviceMemory` and the handle
type (`OPAQUE_FD` or `DMA_BUF`). The memory must have been
allocated with `ExportMemoryAllocateInfo` requesting the
corresponding handle type.

The caller owns the returned fd and must close it when done.
Each call returns a new fd, duplicates are independent.

Linux/Android only. Use `get_memory_win32_handle_khr` on Windows.
