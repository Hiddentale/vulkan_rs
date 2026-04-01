# Usage Notes

Binds buffers for transform feedback output. Each binding slot
receives vertex data streamed from the vertex or geometry shader
during a transform feedback pass.

`first_binding` is the first binding index. Arrays of buffers,
offsets, and sizes specify the output targets.

Must be called before `cmd_begin_transform_feedback_ext`.

Requires `VK_EXT_transform_feedback`.
