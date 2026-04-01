# Usage Notes

Device-address variant of `cmd_begin_conditional_rendering_ext`.
Instead of a buffer handle + offset, the condition is read from a
`DeviceAddress` specified in `ConditionalRenderingBeginInfo2EXT`.

When the 32-bit value at the address is zero, subsequent rendering
and dispatch commands are discarded (or the inverse, if
`INVERTED` is set). End the conditional block with
`cmd_end_conditional_rendering_ext`.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_conditional_rendering`.
