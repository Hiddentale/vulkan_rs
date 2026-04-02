# Usage Notes

Device-address variant of `cmd_copy_image_to_buffer`. Copies image
texel data to a device address range instead of a buffer handle.

The `CopyDeviceMemoryImageInfoKHR` struct specifies the source
image, destination device address, and region descriptions.

This is the device-address counterpart, for the host-side
equivalent, see `copy_image_to_memory` (core 1.4).

Requires `VK_KHR_device_address_commands`.
