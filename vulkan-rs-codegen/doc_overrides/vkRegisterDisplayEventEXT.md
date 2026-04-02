# Usage Notes

Registers a fence to be signaled when a display event occurs.
`DisplayEventInfoEXT` specifies the event type (e.g.,
`FIRST_PIXEL_OUT`, signaled at the start of the first scanline
after a present).

Returns a fence. Useful for frame pacing and display timing.

Requires `VK_EXT_display_control`.
