# Usage Notes

Copies memory regions using indirect parameters stored in a GPU
buffer at the given device address. Enables GPU-driven memory
copy workflows where the copy descriptors are generated on the
device.

Requires `VK_NV_copy_memory_indirect`.
