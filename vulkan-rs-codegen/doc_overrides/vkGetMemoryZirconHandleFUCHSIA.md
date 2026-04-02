# Usage Notes

Exports a Vulkan device memory allocation as a Fuchsia Zircon
VMO handle. The returned handle can be shared with other Fuchsia
processes. Fuchsia OS only.

Requires `VK_FUCHSIA_external_memory`.
