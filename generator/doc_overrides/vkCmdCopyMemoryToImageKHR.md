# Usage Notes

Device-address variant of `cmd_copy_buffer_to_image`. Copies data
from a device address range into an image, instead of reading from
a buffer handle.

The `CopyDeviceMemoryImageInfoKHR` struct specifies the source
device address, destination image, and region descriptions.

This is the device-address counterpart, for the host-side
equivalent, see `copy_memory_to_image` (core 1.4).

Requires `VK_KHR_device_address_commands`.
