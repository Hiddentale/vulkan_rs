# Usage Notes

Dynamically sets the discard rectangles for the current command
buffer. Fragments inside (or outside, depending on mode) these
rectangles are discarded before the fragment shader runs.

Useful for multi-view or split-screen rendering to cheaply cull
fragments that belong to a different viewport.

Requires `VK_EXT_discard_rectangles`.
