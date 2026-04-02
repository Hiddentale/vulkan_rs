# Usage Notes

Dynamically sets the tessellation domain origin:
- `UPPER_LEFT`: Vulkan default, (0,0) is the upper-left corner.
- `LOWER_LEFT`: OpenGL convention, (0,0) is the lower-left.

Affects how tessellation coordinates are interpreted. Useful when
porting OpenGL tessellation shaders.

Provided by `VK_EXT_extended_dynamic_state3`.
