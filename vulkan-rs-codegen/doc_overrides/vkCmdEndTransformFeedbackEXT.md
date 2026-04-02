# Usage Notes

Ends a transform feedback pass started with
`cmd_begin_transform_feedback_ext`. Counter values are written
back to the counter buffers so that a subsequent pass can resume
where this one left off.

Requires `VK_EXT_transform_feedback`.
