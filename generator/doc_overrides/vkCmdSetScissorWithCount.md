# Usage Notes

Dynamically sets both the scissor rectangles and the scissor count.
Only takes effect if the pipeline was created with
`DYNAMIC_STATE_SCISSOR_WITH_COUNT`.

Unlike `cmd_set_scissor` (which requires the count to match the
pipeline's static `viewport_count`), this command also sets the
count dynamically. The scissor count must match the viewport count
set by `cmd_set_viewport_with_count`.

Core dynamic state in Vulkan 1.3.
