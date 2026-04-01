# Usage Notes

Sets per-viewport W scaling factors for lens-matched shading.
The W scaling modifies the clip-space W coordinate to account
for lens distortion in VR headsets, enabling more efficient
shading by varying pixel density across the viewport.

Requires `VK_NV_clip_space_w_scaling`.
