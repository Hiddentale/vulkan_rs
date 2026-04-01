# Usage Notes

Dynamically sets the depth comparison operator. Only takes effect if
the pipeline was created with `DYNAMIC_STATE_DEPTH_COMPARE_OP`.

Values: `COMPARE_OP_LESS` (the common default for forward rendering),
`COMPARE_OP_GREATER` (for reverse-Z), `COMPARE_OP_LESS_OR_EQUAL`,
`COMPARE_OP_ALWAYS`, etc.

**Reverse-Z**: using a reversed depth buffer (near=1.0, far=0.0)
with `COMPARE_OP_GREATER` provides better floating-point precision
distribution across the depth range. This is the recommended setup
for modern rendering.

Core dynamic state in Vulkan 1.3.
