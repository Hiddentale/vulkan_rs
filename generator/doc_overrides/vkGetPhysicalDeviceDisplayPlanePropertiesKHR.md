# Usage Notes

Enumerates display planes supported by a physical device. Each
plane is a compositing layer that can show a portion of a display
surface, multiple planes allow hardware overlay composition.

Each returned `DisplayPlanePropertiesKHR` contains the display
currently connected to the plane and its current stack index.

Use the plane index with `get_display_plane_supported_displays_khr`
to find which displays a given plane can target, and with
`get_display_plane_capabilities_khr` to query positioning and
scaling limits.
