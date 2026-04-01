# Usage Notes

Queries which operations a format supports on this device for
linear tiling, optimal tiling, and buffer usage.

The returned `FormatProperties` contains three `FormatFeatureFlags`
fields:

- **`linear_tiling_features`**: capabilities when the image uses
  `IMAGE_TILING_LINEAR`.
- **`optimal_tiling_features`**: capabilities when the image uses
  `IMAGE_TILING_OPTIMAL` (the common case).
- **`buffer_features`**: capabilities when used in a buffer view
  (e.g. uniform texel buffer, storage texel buffer).

Check the relevant feature bits before creating an image or buffer
view with a particular format. For example, verify
`FORMAT_FEATURE_COLOR_ATTACHMENT` before using a format as a render
target, or `FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR` before
enabling linear filtering.

For extended format properties (Vulkan 1.1+), use
`get_physical_device_format_properties2`.
