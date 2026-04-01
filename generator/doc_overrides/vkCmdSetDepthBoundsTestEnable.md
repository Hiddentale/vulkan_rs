# Usage Notes

Dynamically enables or disables the depth bounds test. Only takes
effect if the pipeline was created with
`DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE`.

When enabled, fragments are discarded if the existing depth buffer
value falls outside the range set by `cmd_set_depth_bounds`. When
disabled, the test is skipped.

Requires the `depth_bounds` device feature.

Core dynamic state in Vulkan 1.3.
