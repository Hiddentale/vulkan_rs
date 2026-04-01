# Usage Notes

Exports a fence's synchronization state as a POSIX file
descriptor. The fd enables cross-process fence synchronization.

`FenceGetFdInfoKHR` specifies the fence and handle type
(`OPAQUE_FD` or `SYNC_FD`). For `SYNC_FD`, the fence must be
signaled or have a pending signal, exporting transfers the
payload and resets the fence.

The caller owns the returned fd and must close it when done.

Linux/Android only. Use `get_fence_win32_handle_khr` on Windows.
