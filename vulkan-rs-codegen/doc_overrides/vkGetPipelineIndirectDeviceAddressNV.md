# Usage Notes

Returns the device address of a compute pipeline's indirect
dispatch metadata. The GPU writes to this address to select
which pipeline to dispatch in device-generated compute workflows.

Requires `VK_NV_device_generated_commands_compute`.
