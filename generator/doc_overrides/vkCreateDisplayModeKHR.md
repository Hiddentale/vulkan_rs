# Usage Notes

Creates a custom display mode with a specific resolution and
refresh rate. Use this when the built-in modes from
`get_display_mode_properties_khr` don't match your requirements.

The `DisplayModeCreateInfoKHR` specifies the visible region
(width/height in pixels) and refresh rate (in millihertz, e.g.,
60000 for 60 Hz).

Not all parameter combinations are valid, the driver may reject
modes it cannot support. If creation fails, fall back to a
built-in mode.
