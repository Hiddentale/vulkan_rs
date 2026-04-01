# Usage Notes

Exports a semaphore's synchronization state as a Windows HANDLE.
The handle can be shared with other processes or APIs for
cross-process GPU synchronization.

`SemaphoreGetWin32HandleInfoKHR` specifies the semaphore and
handle type (`OPAQUE_WIN32`, `OPAQUE_WIN32_KMT`, or
`D3D12_FENCE`).

For `OPAQUE_WIN32` and `D3D12_FENCE`, close the handle with
`CloseHandle` when done. `OPAQUE_WIN32_KMT` handles are
kernel-managed.

Windows only. Use `get_semaphore_fd_khr` on Linux.
