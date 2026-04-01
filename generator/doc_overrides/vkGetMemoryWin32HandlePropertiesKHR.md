# Usage Notes

Queries which memory types are compatible with an external
Windows HANDLE. Use this when importing memory from a handle
received from another process or API (e.g., D3D11 shared
textures).

Returns `MemoryWin32HandlePropertiesKHR` with `memory_type_bits`
indicating compatible memory type indices for import.

Not valid for `OPAQUE_WIN32` or `OPAQUE_WIN32_KMT` handle types, 
those have their memory type determined by the exporting
allocation.
