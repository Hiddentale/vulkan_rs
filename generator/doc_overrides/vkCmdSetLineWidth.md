# Usage Notes

Sets the width of rasterised line primitives dynamically. Only takes
effect if the pipeline was created with `DYNAMIC_STATE_LINE_WIDTH`.

The default line width is 1.0. Wide lines (width > 1.0) require the
`wide_lines` device feature, check
`physical_device_features.wide_lines` before using.

The supported range is device-dependent (query
`physical_device_limits.line_width_range`). If `wide_lines` is not
supported, only 1.0 is valid.

Line width is specified in framebuffer pixels. Anti-aliased lines
may be rendered slightly wider than the specified width due to
coverage calculations.
