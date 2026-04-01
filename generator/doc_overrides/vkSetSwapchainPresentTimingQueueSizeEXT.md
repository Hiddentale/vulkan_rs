# Usage Notes

Sets the maximum number of present timing results the driver
will queue for later retrieval via `get_past_presentation_timing_ext`.
A larger queue prevents timing data from being lost when the
application cannot poll frequently.

Requires `VK_EXT_present_timing`.
