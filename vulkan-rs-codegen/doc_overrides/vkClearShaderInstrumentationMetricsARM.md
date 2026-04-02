# Usage Notes

Resets accumulated metrics for a shader instrumentation object.
Call between frames or profiling sessions to start collecting
fresh data without destroying and recreating the object.

Requires `VK_ARM_shader_instrumentation`.
