# Usage Notes

Retrieves a remote device address for a Vulkan memory allocation,
enabling RDMA (Remote Direct Memory Access) between devices. The
returned address can be used by another device to directly access
this memory over a high-speed interconnect.

Requires `VK_NV_external_memory_rdma`.
