# Usage Notes

Device-address variant of `cmd_begin_transform_feedback_ext`.
Activates transform feedback using counter buffers specified via
device addresses in `BindTransformFeedbackBuffer2InfoEXT` rather
than buffer handles.

`first_counter_range` and the info array identify which transform
feedback counter ranges to resume from. Pass empty counter infos
to start from offset zero.

End the transform feedback pass with
`cmd_end_transform_feedback2_ext`.

Requires `VK_KHR_device_address_commands` and
`VK_EXT_transform_feedback`.
