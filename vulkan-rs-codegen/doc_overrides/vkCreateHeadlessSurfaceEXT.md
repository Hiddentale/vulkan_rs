# Usage Notes

Creates a headless surface that is not associated with any window
system. Useful for off-screen rendering, compute-only workloads,
and automated testing where no display is available.

Destroy with `destroy_surface_khr`.

Requires `VK_EXT_headless_surface`.
