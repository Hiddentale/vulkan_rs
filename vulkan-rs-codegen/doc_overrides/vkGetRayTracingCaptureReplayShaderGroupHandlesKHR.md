# Usage Notes

Retrieves opaque capture/replay handles for shader groups. These
handles allow recreating a ray tracing pipeline with identical
shader group assignments on a subsequent run, enabling
deterministic replay of GPU traces.

Use this for tools, profilers, and capture-replay frameworks.
The handles are passed back via
`RayTracingShaderGroupCreateInfoKHR::shader_group_capture_replay_handle`
when recreating the pipeline.

The pipeline must have been created with
`PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY`.
Requires the `rayTracingPipelineShaderGroupHandleCaptureReplay`
device feature.

The buffer layout is the same as
`get_ray_tracing_shader_group_handles_khr`, sequential handles
of `shaderGroupHandleSize` bytes each.
