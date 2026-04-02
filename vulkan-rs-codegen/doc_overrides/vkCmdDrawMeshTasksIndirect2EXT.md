# Usage Notes

Device-address variant of `cmd_draw_mesh_tasks_indirect_ext`.
Reads mesh shader dispatch parameters from a device address
instead of a buffer handle.

The `DrawIndirect2InfoKHR` specifies the device address, draw
count, and stride.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_mesh_shader`.
