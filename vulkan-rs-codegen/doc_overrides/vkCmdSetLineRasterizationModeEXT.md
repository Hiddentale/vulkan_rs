# Usage Notes

Dynamically sets the line rasterization mode:
- `DEFAULT`: implementation default.
- `RECTANGULAR`: lines are rasterized as parallelograms (Vulkan
  default).
- `BRESENHAM`: pixel-exact Bresenham lines.
- `RECTANGULAR_SMOOTH`: antialiased rectangular lines.

Requires `VK_EXT_line_rasterization` and the corresponding
device feature for the chosen mode.

Provided by `VK_EXT_extended_dynamic_state3`.
