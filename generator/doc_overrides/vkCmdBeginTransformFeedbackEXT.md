# Usage Notes

Begins a transform feedback pass. Vertex shader outputs (or
geometry shader outputs) are written to transform feedback buffers
previously bound with `cmd_bind_transform_feedback_buffers_ext`.

`first_counter_buffer` and `p_counter_buffer_offsets` specify
where to resume writing from (pass null offsets to start from
scratch).

End with `cmd_end_transform_feedback_ext`.

Requires `VK_EXT_transform_feedback` and the
`transformFeedback` feature.
