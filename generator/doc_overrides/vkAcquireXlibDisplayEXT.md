# Usage Notes

Acquires exclusive control of an X11 display for direct rendering,
bypassing the X server's compositor. The display must be released
with `release_display_ext` when finished.

Requires `VK_EXT_acquire_xlib_display`. Linux/X11 only.
