# Usage Notes

Ends shader instrumentation collection in a command buffer.
Metrics collected since the matching
`cmd_begin_shader_instrumentation_arm` can be retrieved with
`get_shader_instrumentation_values_arm` after submission
completes.

Requires `VK_ARM_shader_instrumentation`.
