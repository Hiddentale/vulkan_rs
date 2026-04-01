# Usage Notes

Creates an opacity micromap object. Micromaps store per-triangle
opacity or hit data at sub-triangle granularity, enabling the
ray tracing implementation to skip fully transparent micro-
triangles without invoking any-hit shaders.

The `MicromapCreateInfoEXT` specifies the backing buffer, size,
and type (`OPACITY_MICROMAP`).

Build with `cmd_build_micromaps_ext` or `build_micromaps_ext`.
Destroy with `destroy_micromap_ext`.

Requires `VK_EXT_opacity_micromap`.
