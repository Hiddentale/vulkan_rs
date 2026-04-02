# Usage Notes

Sets custom sample locations for multisampled rasterization.
`SampleLocationsInfoEXT` specifies the grid size, sample count,
and per-sample (x, y) positions within each pixel.

Custom sample locations enable techniques like temporal AA
jittering and programmable MSAA patterns.

Must be called when custom sample locations are enabled (via
`cmd_set_sample_locations_enable_ext` or pipeline state).

Requires `VK_EXT_sample_locations`.
