# Usage Notes

Dispatches a compute shader with a non-zero base workgroup offset.
The shader's `gl_WorkGroupID` starts at (`base_group_x`,
`base_group_y`, `base_group_z`) instead of (0, 0, 0).

This is useful for splitting a large dispatch across multiple
submissions or for device groups where different physical devices
handle different regions of the workgroup space.

`gl_NumWorkGroups` reflects the `group_count` parameters, not the
total. The shader sees workgroup IDs in the range
[`base`, `base + count`).

For single-GPU, zero-base dispatches, use `cmd_dispatch` instead.
