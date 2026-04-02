# Usage Notes

Dynamically sets the number of control points per patch for
tessellation. Overrides the value specified at pipeline creation.

Typical values: 3 (triangles), 4 (quads), 16 (bicubic patches).
The maximum is `maxTessellationPatchSize` (at least 32).

Requires the `extendedDynamicState2PatchControlPoints` feature.

Provided by `VK_EXT_extended_dynamic_state2`.
