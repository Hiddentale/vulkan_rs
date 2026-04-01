# Usage Notes

Exports a device memory allocation as a Windows HANDLE. The
handle can be shared with other processes or APIs (D3D12, CUDA)
for GPU memory interop.

`MemoryGetWin32HandleInfoKHR` specifies the `DeviceMemory` and
handle type (`OPAQUE_WIN32` or `OPAQUE_WIN32_KMT`).

For `OPAQUE_WIN32`, the handle must be closed with `CloseHandle`
when done. `OPAQUE_WIN32_KMT` handles are kernel-managed and do
not need explicit cleanup.

Windows only. Use `get_memory_fd_khr` on Linux.
