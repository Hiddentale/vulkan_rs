# Usage Notes

Sets the depth bounds test range dynamically. Only takes effect if
the pipeline was created with `DYNAMIC_STATE_DEPTH_BOUNDS` and
depth bounds testing is enabled in the depth-stencil state.

The depth bounds test discards fragments whose depth buffer value
falls outside [`min_depth_bounds`, `max_depth_bounds`]. Note that
this tests the **existing** depth buffer value, not the fragment's
incoming depth.

Use cases are niche:

- **Stencil shadow volumes**: reject fragments that are clearly
  outside the shadow volume's depth range.
- **Deferred shading light volumes**: skip fragments outside the
  light's depth range.

Requires the `depth_bounds` device feature. Not supported on all
hardware, check `physical_device_features.depth_bounds` before
enabling.
