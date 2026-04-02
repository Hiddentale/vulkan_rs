# Usage Notes

Vulkan 1.4 version of `cmd_bind_index_buffer` that additionally
accepts a `size` parameter specifying the valid range of the index
buffer. This enables the driver to perform bounds checking.

Pass `VK_WHOLE_SIZE` for the size to use the remainder of the buffer
from the offset.

Prefer this over `cmd_bind_index_buffer` when targeting Vulkan 1.4+.
