# Usage Notes

Dynamically sets both the viewports and the viewport count. Only
takes effect if the pipeline was created with
`DYNAMIC_STATE_VIEWPORT_WITH_COUNT`.

Unlike `cmd_set_viewport` (which requires the count to match the
pipeline's static `viewport_count`), this command also sets the
count dynamically. The viewport count must match the scissor count
set by `cmd_set_scissor_with_count`.

Core dynamic state in Vulkan 1.3.
