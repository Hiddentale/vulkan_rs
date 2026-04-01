# Usage Notes

Decompresses data from one memory region into another on the GPU.
The decompression algorithm is specified in the info structure.

Useful for loading compressed assets directly on the GPU without
a CPU round-trip.

Requires `VK_EXT_memory_decompression`.
