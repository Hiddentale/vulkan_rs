# Usage Notes

Queries the image formats compatible with a video profile for
decoded output, DPB reference, or encode input images.

Specify the intended usage in `PhysicalDeviceVideoFormatInfoKHR`
(image usage flags indicating decode output, DPB, or encode
input). The returned `VideoFormatPropertiesKHR` list the
compatible formats, image types, tiling modes, and usage flags.

Use these results to create images that are compatible with the
video session. Using an unsupported format results in validation
errors.
