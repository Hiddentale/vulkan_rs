# Usage Notes

Dynamically sets the depth clamp range. When depth clamping is
enabled, fragments are clamped to the specified min/max depth
values instead of the viewport near/far range.

Pass null to use the default viewport depth range for clamping.

Requires `VK_EXT_depth_clamp_control` and the
`depthClampControl` feature.
