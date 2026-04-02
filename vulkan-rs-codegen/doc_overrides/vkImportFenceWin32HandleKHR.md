# Usage Notes

Imports a synchronization payload from a Windows HANDLE into a
fence.

`ImportFenceWin32HandleInfoKHR` specifies the target fence,
handle type, handle (or name), and whether the import is
temporary or permanent.

The handle is duplicated internally, the caller retains
ownership and should close it when no longer needed.

Windows only. Use `import_fence_fd_khr` on Linux.
