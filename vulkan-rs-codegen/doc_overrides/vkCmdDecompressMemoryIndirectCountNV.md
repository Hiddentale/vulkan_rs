# Usage Notes

Indirect-count variant of `cmd_decompress_memory_nv`. Reads the
decompression region descriptors and count from GPU buffer
addresses, enabling fully GPU-driven decompression.

Requires `VK_NV_memory_decompression`.
