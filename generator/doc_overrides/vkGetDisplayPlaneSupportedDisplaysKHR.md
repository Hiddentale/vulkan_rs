# Usage Notes

Returns the list of displays that a given display plane can be
used with. Not all planes can target all displays, this query
lets you find valid plane-display pairings.

`plane_index` is an index into the array returned by
`get_physical_device_display_plane_properties_khr`.

After finding a compatible display, query its modes with
`get_display_mode_properties_khr` and configure the plane via
`DisplaySurfaceCreateInfoKHR`.
