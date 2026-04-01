# Usage Notes

Device-address variant of `cmd_update_buffer`. Writes a small
amount of inline data to a device address range, instead of
targeting a buffer handle.

`data_size` must be ≤ 65536 bytes and a multiple of 4. For
larger transfers, use `cmd_copy_memory_khr`.

The `DeviceAddressRangeKHR` specifies the destination address.

Requires `VK_KHR_device_address_commands`.
