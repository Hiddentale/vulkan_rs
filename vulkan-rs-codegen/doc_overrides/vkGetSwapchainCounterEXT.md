# Usage Notes

Queries a performance counter associated with a swapchain (e.g.,
vertical blanks). Returns the counter value as a `u64`.

The counter type is specified as a `SurfaceCounterFlagBitsEXT`
(typically `VBLANK`).

Requires `VK_EXT_display_control`.
