# Usage Notes

Device-address variant of the AMD buffer marker extension. Writes
a 32-bit marker value to a device address at a specific pipeline
stage.

The `MemoryMarkerInfoAMD` specifies the pipeline stage, device
address, and marker value. Useful for GPU crash debugging, 
markers that were written indicate how far the GPU progressed
before a fault.

Requires `VK_KHR_device_address_commands`. This is the
device-address counterpart of `cmd_write_buffer_marker_amd`.
