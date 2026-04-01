# Usage Notes

Dynamically sets the polygon rasterization mode:
- `FILL`: normal filled triangles (default).
- `LINE`: wireframe rendering.
- `POINT`: vertices only.

`LINE` and `POINT` require the `fillModeNonSolid` device feature.

Provided by `VK_EXT_extended_dynamic_state3`.
