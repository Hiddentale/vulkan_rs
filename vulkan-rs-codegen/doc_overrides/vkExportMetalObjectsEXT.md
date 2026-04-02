# Usage Notes

Exports the underlying Metal objects (device, command queue,
textures, buffers, etc.) from Vulkan objects. Chain the
appropriate export struct into the pNext of the info structure
to select which Metal object to retrieve.

Useful for Metal interop on Apple platforms where both APIs
share the same GPU resources.

Requires `VK_EXT_metal_objects`. macOS/iOS only.
