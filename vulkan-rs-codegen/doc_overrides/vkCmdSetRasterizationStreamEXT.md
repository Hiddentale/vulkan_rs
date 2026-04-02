# Usage Notes

Dynamically sets which vertex stream is used for rasterization
when transform feedback is active. Stream 0 is always rasterized
by default; other streams can be selected with this command.

Requires `VK_EXT_transform_feedback` and the
`geometryStreams` feature.

Provided by `VK_EXT_extended_dynamic_state3`.
