# Usage Notes

Vulkan 1.3 version of `cmd_blit_image` that uses an extensible
`BlitImageInfo2` struct.

Chain `BlitImageCubicWeightsInfoQCOM` for cubic filtering on
Qualcomm hardware. Otherwise functionally identical to
`cmd_blit_image`.

Prefer this when targeting Vulkan 1.3+.
