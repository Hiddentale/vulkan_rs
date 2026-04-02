# Usage Notes

Queries whether a buffer with the given usage and flags can be
exported to or imported from an external handle type (e.g. POSIX
file descriptor, Win32 handle, or DMA-BUF).

The returned `ExternalBufferProperties` indicates:

- Whether the external handle type is compatible.
- Whether dedicated allocation is required.
- Which other handle types the memory can be exported to
  simultaneously.

Use this before creating a buffer intended for cross-process or
cross-API sharing to verify the external memory capabilities.
