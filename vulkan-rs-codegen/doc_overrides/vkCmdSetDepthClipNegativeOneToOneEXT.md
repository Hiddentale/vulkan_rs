# Usage Notes

Dynamically sets the depth clip range convention:
- `true`: NDC depth range is [-1, 1] (OpenGL convention).
- `false`: NDC depth range is [0, 1] (Vulkan default).

Useful for porting OpenGL content or using reversed-Z with
OpenGL-style projections.

Requires `VK_EXT_depth_clip_control` and the
`depthClipControl` feature.

Provided by `VK_EXT_extended_dynamic_state3`.
