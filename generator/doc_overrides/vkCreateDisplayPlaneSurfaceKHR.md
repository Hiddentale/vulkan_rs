# Usage Notes

Creates a `SurfaceKHR` for direct display output, bypassing the
window system. The surface is presented directly on a display
plane.

Configure via `DisplaySurfaceCreateInfoKHR`:

- `display_mode`: a mode from `get_display_mode_properties_khr`
  or `create_display_mode_khr`.
- `plane_index`: which display plane to use.
- `plane_stack_index`: z-ordering among planes.
- `transform`: rotation/mirroring.
- `alpha_mode`: how alpha is handled (opaque, global, per-pixel).
- `image_extent`: the surface resolution.

After creation, use the surface with `create_swapchain_khr` like
any other surface. Destroy with `destroy_surface_khr`.
