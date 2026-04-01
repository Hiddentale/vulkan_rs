# Usage Notes

Exports a semaphore's synchronization state as a POSIX file
descriptor. The fd can be transferred to another process for
cross-process GPU synchronization.

`SemaphoreGetFdInfoKHR` specifies the semaphore and handle type
(`OPAQUE_FD` or `SYNC_FD`). For `SYNC_FD`, the semaphore must
be signaled or have a pending signal operation, exporting
transfers ownership and resets the semaphore to unsignaled.

The caller owns the returned fd. For `OPAQUE_FD`, each export
creates a new reference. For `SYNC_FD`, the export is a
move, the semaphore payload is transferred.

Linux/Android only. Use `get_semaphore_win32_handle_khr` on
Windows.
