# Usage Notes

Dynamically sets the stencil operations for front-facing,
back-facing, or both face sets. Only takes effect if the pipeline
was created with `DYNAMIC_STATE_STENCIL_OP`.

Sets four values per face:

- **`fail_op`**: action when the stencil test fails.
- **`pass_op`**: action when both stencil and depth tests pass.
- **`depth_fail_op`**: action when stencil passes but depth fails.
- **`compare_op`**: the stencil comparison function.

Common operations: `KEEP`, `REPLACE`, `INCREMENT_AND_CLAMP`,
`DECREMENT_AND_CLAMP`, `INVERT`, `ZERO`.

Core dynamic state in Vulkan 1.3.
