# Usage Notes

Enumerates the display modes supported by a display. Each
`DisplayModePropertiesKHR` contains a mode handle and its
parameters (visible region resolution and refresh rate).

Use these to select an appropriate mode for
`DisplaySurfaceCreateInfoKHR`, or create a custom mode with
`create_display_mode_khr` if the desired parameters are not
listed.

Prefer `get_display_mode_properties2_khr` when
`VK_KHR_get_display_properties2` is available.
