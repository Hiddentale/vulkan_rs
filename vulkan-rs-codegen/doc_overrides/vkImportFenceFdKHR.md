# Usage Notes

Imports a synchronization payload from a POSIX file descriptor
into a fence.

`ImportFenceFdInfoKHR` specifies the target fence, handle type,
fd, and whether the import is temporary (payload consumed on
first wait/reset) or permanent.

For `SYNC_FD`, ownership of the fd transfers to the
implementation, do not close it. For `OPAQUE_FD`, the fd is
duplicated and can be closed after the call.

Linux/Android only. Use `import_fence_win32_handle_khr` on
Windows.
