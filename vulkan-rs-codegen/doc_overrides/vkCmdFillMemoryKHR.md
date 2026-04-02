# Usage Notes

Device-address variant of `cmd_fill_buffer`. Fills a device
address range with a repeating 32-bit `data` value, instead of
targeting a buffer handle.

The `DeviceAddressRangeKHR` specifies the destination address
and size. The fill size must be a multiple of 4 bytes.

Requires `VK_KHR_device_address_commands`.
