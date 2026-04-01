# Usage Notes

Queries the memory requirements for storing a compute pipeline's
indirect dispatch metadata. Allocate a buffer of this size and
pass it when creating the pipeline for device-generated compute
dispatch.

Requires `VK_NV_device_generated_commands_compute`.
