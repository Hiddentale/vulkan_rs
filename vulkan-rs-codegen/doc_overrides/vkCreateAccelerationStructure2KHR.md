# Usage Notes

Device-address variant of `create_acceleration_structure_khr`.
Creates a ray tracing acceleration structure backed by a device
address range instead of a buffer handle.

The `AccelerationStructureCreateInfo2KHR` specifies the device
address, size, type (top-level or bottom-level), and optional
capture/replay address.

Destroy with `destroy_acceleration_structure_khr`.

Requires `VK_KHR_device_address_commands` and
`VK_KHR_acceleration_structure`.
