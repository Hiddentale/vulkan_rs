# Usage Notes

Dynamically enables or disables depth clipping. When disabled,
primitives are not clipped against the near and far planes
(equivalent to `depthClampEnable`, but controlled separately).

Requires `VK_EXT_depth_clip_enable` and the `depthClipEnable`
feature.

Provided by `VK_EXT_extended_dynamic_state3`.
