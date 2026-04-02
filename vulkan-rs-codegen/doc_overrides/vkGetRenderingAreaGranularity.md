# Usage Notes

Vulkan 1.4 command that queries the optimal render area granularity
for a given dynamic rendering configuration, without needing a
render pass object.

This is the dynamic rendering equivalent of
`get_render_area_granularity`. Pass a `RenderingAreaInfo` describing
the attachment formats and sample count, and receive the granularity
(in pixels) at which the render area should be aligned.

On most desktop GPUs this returns (1, 1). On tile-based GPUs the
granularity may match the tile size.
