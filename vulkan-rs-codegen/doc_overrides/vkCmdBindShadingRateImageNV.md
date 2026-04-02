# Usage Notes

Binds a shading rate image that controls per-region fragment
shading rate. Each texel in the image maps to a tile of the
framebuffer and specifies the coarse shading rate for that tile.

Requires `VK_NV_shading_rate_image`.
