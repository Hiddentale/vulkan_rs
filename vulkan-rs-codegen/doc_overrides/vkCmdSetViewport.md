# Usage Notes

Sets the viewport transform dynamically. Only takes effect if the
pipeline was created with `DYNAMIC_STATE_VIEWPORT`.

The viewport defines the mapping from normalised device coordinates
to framebuffer coordinates. Most applications use a single viewport
covering the full render target:

```text
Viewport { x: 0.0, y: 0.0, width: w, height: h, min_depth: 0.0, max_depth: 1.0 }
```

**Flipped Y**: Vulkan's clip space has Y pointing downward (unlike
OpenGL). To match OpenGL conventions, use a negative height and
offset Y by the framebuffer height:
`Viewport { y: h, height: -h, ... }`. This requires Vulkan 1.1 or
`VK_KHR_maintenance1`.

Multiple viewports are supported for multi-view rendering (e.g.
VR). The `first_viewport` parameter selects which viewport index to
start writing at.
