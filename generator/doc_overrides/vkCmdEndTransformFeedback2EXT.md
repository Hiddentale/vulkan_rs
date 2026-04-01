# Usage Notes

Device-address variant of `cmd_end_transform_feedback_ext`.
Stops transform feedback and writes counter values to device
addresses specified in `BindTransformFeedbackBuffer2InfoEXT`.

These saved counter values can be passed to
`cmd_begin_transform_feedback2_ext` to resume feedback in a
later render pass.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_transform_feedback`.
