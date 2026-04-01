# Usage Notes

Writes a 32-bit marker value into a buffer after a specified
pipeline stage completes. Useful for fine-grained GPU progress
tracking and debugging GPU hangs by identifying the last
completed stage.

Requires `VK_AMD_buffer_marker`.
