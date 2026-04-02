# Usage Notes

Dynamically sets the extra overestimation size for conservative
rasterization. This expands the primitive by additional pixels
beyond the minimum overestimation guaranteed by the implementation.

The value is in units of pixels. 0.0 means no extra
overestimation beyond the implementation minimum.

Requires `VK_EXT_conservative_rasterization`.

Provided by `VK_EXT_extended_dynamic_state3`.
