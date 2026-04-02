# Usage Notes

Device-address variant of `cmd_draw_indexed_indirect`. Reads
indexed draw parameters from a device address instead of a buffer
handle + offset.

The `DrawIndirect2InfoKHR` specifies the device address, draw
count, and stride.

Requires `VK_KHR_device_address_commands`.
