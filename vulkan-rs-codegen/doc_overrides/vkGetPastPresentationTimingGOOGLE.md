# Usage Notes

Returns timing data for past presentations on a swapchain,
including actual present time, earliest possible present time,
and finish time. Uses the two-call idiom (call once to get the
count, again to fill the buffer). Useful for frame pacing and
latency analysis.

Requires `VK_GOOGLE_display_timing`.
