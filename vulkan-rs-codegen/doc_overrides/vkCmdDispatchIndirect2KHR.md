# Usage Notes

Device-address variant of `cmd_dispatch_indirect`. Reads the
dispatch parameters (group counts) from a device address instead
of a buffer handle + offset.

The `DispatchIndirect2InfoKHR` specifies the device address where
the `DispatchIndirectCommand` struct resides.

Requires `VK_KHR_device_address_commands`.
