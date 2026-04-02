# Usage Notes

Extended version of `cmd_set_depth_bias` that takes a
`DepthBiasInfoEXT` struct with pNext extensibility. This allows
chaining `DepthBiasRepresentationInfoEXT` to control the depth
bias representation (least-representable-value vs. float).

Requires `VK_EXT_depth_bias_control`.
