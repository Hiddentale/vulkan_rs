# Usage Notes

Dynamically sets the line stipple pattern and repeat factor. Only
takes effect if the pipeline was created with
`DYNAMIC_STATE_LINE_STIPPLE`.

The `stipple_factor` (1–256) controls how many pixels each bit of
the pattern spans. The `stipple_pattern` is a 16-bit bitmask where
each bit represents a pixel, 1 is drawn, 0 is discarded.

Line stippling requires `VK_EXT_line_rasterization` and the
`stippled_*_lines` device features, depending on which line
rasterisation mode you use.

Core dynamic state in Vulkan 1.4.
