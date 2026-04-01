# Usage Notes

Copies memory regions using parameters read from a device-side
info structure. Enables GPU-driven memory copies without CPU
involvement in specifying source/destination addresses.

Requires `VK_KHR_copy_memory_indirect`.
