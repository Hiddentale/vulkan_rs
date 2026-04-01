# Usage Notes

Indirect variant of `cmd_decompress_memory_ext`. Reads the
decompression parameters and count from GPU-visible buffer
addresses, enabling fully GPU-driven decompression workflows.

Requires `VK_EXT_memory_decompression`.
