# Usage Notes

Queries the capabilities of a display plane when used with a
specific display mode. The returned `DisplayPlaneCapabilitiesKHR`
describes:

- Supported alpha modes (opaque, global, per-pixel).
- Min/max source and destination extents, the range of scaling
  the plane supports.
- Min/max source and destination positions, how the plane
  image can be positioned.

Use these limits to configure `DisplaySurfaceCreateInfoKHR`
correctly. Exceeding the reported limits results in validation
errors.
