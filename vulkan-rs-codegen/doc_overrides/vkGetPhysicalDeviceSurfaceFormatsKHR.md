# Usage Notes

Returns the list of supported format + colour space pairs for a
surface. Pick one of these pairs for swapchain creation.

The most portable choice is `FORMAT_B8G8R8A8_SRGB` +
`COLOR_SPACE_SRGB_NONLINEAR`. If your preferred format is not
listed, fall back to the first available pair.

If the list contains a single entry with `FORMAT_UNDEFINED`, the
surface has no preferred format and any format is acceptable.

For HDR output, look for `COLOR_SPACE_HDR10_ST2084_EXT` or similar
extended colour spaces.
