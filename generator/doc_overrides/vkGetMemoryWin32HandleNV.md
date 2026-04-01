# Usage Notes

Exports a Vulkan device memory allocation as a Win32 handle
(HANDLE) for sharing with other APIs or processes. This is the
legacy NV path; prefer `get_memory_win32_handle_khr` for new
code.

Requires `VK_NV_external_memory_win32`. Windows only.
