# Usage Notes

Sets the device mask for subsequent commands in a command buffer
when using device groups (multi-GPU with
`VK_KHR_device_group` / Vulkan 1.1).

The device mask is a bitmask where bit *i* indicates that subsequent
commands execute on physical device *i* in the device group.

For single-GPU systems (the common case), the device mask is always
`0x1` and this command is not needed.

Must only be called outside a render pass, or inside a render pass
that was begun with `DEVICE_GROUP_BEGIN_INFO` and has the
`DEVICE_GROUP` flag set.
