# Usage Notes

Imports a synchronization payload from a POSIX file descriptor
into a semaphore. After import, the semaphore uses the external
synchronization state.

`ImportSemaphoreFdInfoKHR` specifies the target semaphore, handle
type, fd, and whether the import is temporary (payload consumed
on first wait) or permanent.

For `SYNC_FD`, the import takes ownership of the fd, do not
close it afterward. For `OPAQUE_FD`, the fd is duplicated
internally and can be closed after the call.

Linux/Android only. Use `import_semaphore_win32_handle_khr` on
Windows.
