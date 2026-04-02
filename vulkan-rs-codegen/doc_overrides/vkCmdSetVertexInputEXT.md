# Usage Notes

Dynamically sets the complete vertex input state: bindings
(stride, input rate) and attributes (location, format, offset).

Replaces `VertexInputBindingDescription` and
`VertexInputAttributeDescription` from pipeline creation. Pass
arrays of `VertexInputBindingDescription2EXT` and
`VertexInputAttributeDescription2EXT`.

Provided by `VK_EXT_vertex_input_dynamic_state`.
