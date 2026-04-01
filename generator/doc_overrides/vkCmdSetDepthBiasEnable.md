# Usage Notes

Dynamically enables or disables depth bias. Only takes effect if the
pipeline was created with `DYNAMIC_STATE_DEPTH_BIAS_ENABLE`.

When enabled, the depth bias values set by `cmd_set_depth_bias` are
applied to fragment depth values. When disabled, no bias is applied
regardless of the bias values.

Useful for toggling shadow map bias without separate pipelines.

Core dynamic state in Vulkan 1.3.
