# Usage Notes

Begins an indexed query, like `cmd_begin_query` but with an
additional `index` parameter that selects which vertex stream
to query when used with transform feedback statistics queries.

For `QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT`, the index
selects the stream (0–3). For other query types, index must be 0.

End with `cmd_end_query_indexed_ext`.

Requires `VK_EXT_transform_feedback`.
