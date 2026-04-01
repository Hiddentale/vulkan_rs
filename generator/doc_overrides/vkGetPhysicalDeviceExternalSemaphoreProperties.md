# Usage Notes

Queries whether a semaphore can be exported to or imported from an
external handle type (e.g. POSIX file descriptor, Win32 handle,
sync file, or Zircon event).

The returned properties indicate:

- Whether the external handle type is compatible with semaphores.
- Whether the handle is a copy or a reference.
- Which other handle types the semaphore can be exported to.

External semaphores are the primary cross-process and cross-API
synchronisation mechanism, for example, synchronising Vulkan
rendering with an OpenGL or DirectX consumer.
