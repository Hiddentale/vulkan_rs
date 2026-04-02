# Usage Notes

Records a GPU-side micromap build into a command buffer. Each
`MicromapBuildInfoEXT` specifies the source triangle opacity
data, destination micromap, and scratch memory.

For the CPU-side equivalent, see `build_micromaps_ext`.

Requires `VK_EXT_opacity_micromap`.
