# Usage Notes

Creates a shader instrumentation object that configures which
metrics to collect during shader execution. The instrumentation
is later bound to command buffers with
`cmd_begin_shader_instrumentation_arm`.

Requires `VK_ARM_shader_instrumentation`.
