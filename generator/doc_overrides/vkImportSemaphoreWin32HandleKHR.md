# Usage Notes

Imports a synchronization payload from a Windows HANDLE into a
semaphore. After import, the semaphore uses the external
synchronization state.

`ImportSemaphoreWin32HandleInfoKHR` specifies the target
semaphore, handle type, handle (or name for named handles), and
whether the import is temporary or permanent.

The handle is duplicated internally, the caller retains
ownership and should close it when no longer needed.

Windows only. Use `import_semaphore_fd_khr` on Linux.
