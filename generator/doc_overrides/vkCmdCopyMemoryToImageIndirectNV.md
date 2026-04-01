# Usage Notes

Copies data from memory to an image using indirect parameters
stored at a device address. Enables GPU-driven texture uploads
where the copy regions are generated on the device.

Requires `VK_NV_copy_memory_indirect`.
