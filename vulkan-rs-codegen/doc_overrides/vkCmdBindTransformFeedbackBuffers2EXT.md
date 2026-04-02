# Usage Notes

Device-address variant of `cmd_bind_transform_feedback_buffers_ext`.
Binds transform feedback output buffers using device addresses
instead of buffer handles.

Each `BindTransformFeedbackBuffer2InfoEXT` specifies a device
address and size for one binding slot starting at `first_binding`.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_transform_feedback`.
