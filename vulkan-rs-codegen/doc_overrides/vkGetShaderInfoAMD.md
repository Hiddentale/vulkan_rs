# Usage Notes

Queries AMD-specific shader information such as compiled binary
statistics, disassembly, or resource usage. Call once with a null
buffer to query the size, then again with an appropriately sized
buffer.

Requires `VK_AMD_shader_info`.
