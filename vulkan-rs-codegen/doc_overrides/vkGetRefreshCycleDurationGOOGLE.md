# Usage Notes

Queries the duration of a single refresh cycle (vsync interval)
for the display associated with the given swapchain. Returns
the period in nanoseconds. Essential for accurate frame pacing.

Requires `VK_GOOGLE_display_timing`.
