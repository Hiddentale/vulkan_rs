# Usage Notes

Attaches arbitrary binary data to a Vulkan object. Unlike
`set_debug_utils_object_name_ext` (which sets a string),
tags carry opaque byte data identified by a `tag_name` (u64).

Tags are consumed by debugging tools and layers that understand
the specific `tag_name` value. Most applications only need
`set_debug_utils_object_name_ext`, use tags for tool-specific
metadata.

Requires `VK_EXT_debug_utils`.
