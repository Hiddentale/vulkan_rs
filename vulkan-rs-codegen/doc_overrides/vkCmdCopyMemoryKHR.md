# Usage Notes

Copies data between two device address ranges on the GPU. This is
the device-address equivalent of `cmd_copy_buffer`, instead of
buffer handles, source and destination are specified as device
addresses in `CopyDeviceMemoryInfoKHR`.

Useful for copying data between arbitrary device memory locations
without needing buffer objects.

Requires `VK_KHR_device_address_commands`.
