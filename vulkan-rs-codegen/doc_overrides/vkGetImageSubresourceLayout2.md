# Usage Notes

Vulkan 1.4 version of `get_image_subresource_layout` that uses
extensible structs via pNext.

Returns the layout (offset, size, row pitch, array pitch, depth
pitch) for a given subresource of an existing image. Chain
`ImageCompressionPropertiesEXT` to query fixed-rate compression
state.

For linear-tiling images, this tells you how to access texels
through a mapped pointer. For optimal-tiling images, the layout is
opaque and implementation-defined.
