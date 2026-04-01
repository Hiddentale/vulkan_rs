# Usage Notes

Returns the granularity (in pixels) at which the render area should
be aligned for optimal performance with this render pass.

The render area passed to `cmd_begin_render_pass` should have its
`offset` be a multiple of this granularity and its `extent` should
either cover the full framebuffer or be rounded up to a multiple.

On most desktop GPUs this returns (1, 1), meaning any alignment is
fine. On tile-based GPUs this may return the tile size (e.g. 32×32),
and misalignment can cause partial-tile overhead.

In practice, most applications render to the full framebuffer extent
and never need to worry about this.
