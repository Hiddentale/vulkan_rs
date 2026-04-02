# Usage Notes

Begins shader instrumentation collection in a command buffer.
All subsequent draw and dispatch commands will collect the
metrics configured in the instrumentation object until
`cmd_end_shader_instrumentation_arm` is called.

Requires `VK_ARM_shader_instrumentation`.
