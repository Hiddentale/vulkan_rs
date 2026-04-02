# Usage Notes

Dynamically enables or disables primitive restart. Only takes effect
if the pipeline was created with
`DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE`.

When enabled, a special index value (0xFFFF for UINT16, 0xFFFFFFFF
for UINT32) in the index buffer ends the current primitive and
starts a new one. This lets you draw multiple triangle strips or
line strips from a single draw call without degenerate triangles.

Only meaningful for strip topologies (`TRIANGLE_STRIP`,
`LINE_STRIP`, `TRIANGLE_FAN`, etc.).

Core dynamic state in Vulkan 1.3.
