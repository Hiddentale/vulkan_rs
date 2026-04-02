# Usage Notes

Queries the memory properties (compatible memory type bits) for
a Zircon VMO handle. Use before importing external memory to
determine which memory type to allocate. Fuchsia OS only.

Requires `VK_FUCHSIA_external_memory`.
