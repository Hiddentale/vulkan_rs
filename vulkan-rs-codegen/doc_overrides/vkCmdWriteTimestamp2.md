# Usage Notes

Vulkan 1.3 version of `cmd_write_timestamp` that uses 64-bit
pipeline stage flags (`PipelineStageFlags2`).

The wider stage flags support newer stages like
`PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR` and
`PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT` that do not fit in the
original 32-bit field.

Prefer this over `cmd_write_timestamp` when targeting Vulkan 1.3+.
