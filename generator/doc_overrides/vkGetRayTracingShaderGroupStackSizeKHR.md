# Usage Notes

Queries the stack size contribution of a single shader within a
shader group. The result is used to compute the total pipeline
stack size for `cmd_set_ray_tracing_pipeline_stack_size_khr`.

`group` indexes into the shader groups array from pipeline
creation. `group_shader` selects which shader within the group:
`GENERAL`, `CLOSEST_HIT`, `ANY_HIT`, or `INTERSECTION`.

The default pipeline stack size is computed automatically at
creation time, but it assumes worst-case recursion. If you know
your actual `maxPipelineRayRecursionDepth` is lower, query
individual stack sizes and compute a tighter total to reduce
scratch memory usage.

Stack size computation formula (from spec):

`raygen + max(closesthit + intersection, miss, callable) * maxRecursionDepth`

Call this per-shader, aggregate across all groups, then set the
result with `cmd_set_ray_tracing_pipeline_stack_size_khr`.
