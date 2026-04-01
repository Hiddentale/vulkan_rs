# Usage Notes

Dynamically sets the conservative rasterization mode:
- `DISABLED`: normal rasterization.
- `OVERESTIMATE`: a fragment is generated if the primitive
  overlaps any part of the pixel.
- `UNDERESTIMATE`: a fragment is generated only if the pixel
  is fully covered.

Requires `VK_EXT_conservative_rasterization`.

Provided by `VK_EXT_extended_dynamic_state3`.
