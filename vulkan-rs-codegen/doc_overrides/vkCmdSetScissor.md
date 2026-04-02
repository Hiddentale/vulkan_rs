# Usage Notes

Sets the scissor rectangle dynamically. Only takes effect if the
pipeline was created with `DYNAMIC_STATE_SCISSOR`.

The scissor test discards fragments outside the rectangle. Unlike
the viewport (which transforms coordinates), the scissor is a
hard clip in framebuffer pixel coordinates.

A common default is a scissor covering the full framebuffer:

```text
Rect2D { offset: { x: 0, y: 0 }, extent: { width: w, height: h } }
```

Scissor rectangles are useful for UI rendering, split-screen, or
any case where you want to restrict rendering to a sub-region
without changing the viewport transform.

The scissor must be set before any draw call when using dynamic
scissor state, even if it covers the full framebuffer.
