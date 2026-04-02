# Usage Notes

Begins a conditional rendering block. Subsequent rendering and
dispatch commands are discarded if the 32-bit value at the
specified buffer offset is zero (or non-zero if `INVERTED` is
set).

End with `cmd_end_conditional_rendering_ext`.

Useful for GPU-driven occlusion culling, write visibility
results to a buffer, then conditionally skip draw calls.

Requires `VK_EXT_conditional_rendering`.
