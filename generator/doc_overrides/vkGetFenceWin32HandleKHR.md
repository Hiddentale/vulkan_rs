# Usage Notes

Exports a fence's synchronization state as a Windows HANDLE
for cross-process fence synchronization.

`FenceGetWin32HandleInfoKHR` specifies the fence and handle type
(`OPAQUE_WIN32` or `OPAQUE_WIN32_KMT`).

For `OPAQUE_WIN32`, close the handle with `CloseHandle` when
done. `OPAQUE_WIN32_KMT` handles are kernel-managed.

Windows only. Use `get_fence_fd_khr` on Linux.
