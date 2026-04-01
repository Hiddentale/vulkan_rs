# Usage Notes

Draws vertices using a byte count stored in a counter buffer
(typically written by a transform feedback pass). The vertex
count is computed as `counter_value / vertex_stride`.

`counter_offset` accounts for any header bytes before the
counter value in the buffer.

Requires `VK_EXT_transform_feedback`.
