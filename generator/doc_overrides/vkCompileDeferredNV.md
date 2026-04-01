# Usage Notes

Triggers compilation of a deferred shader in an NV ray tracing
pipeline. When a pipeline is created with
`PIPELINE_CREATE_DEFER_COMPILE_BIT_NV`, individual shaders can
be compiled on demand using this command.

Useful for spreading compilation cost across frames or threads.

Requires `VK_NV_ray_tracing`.
