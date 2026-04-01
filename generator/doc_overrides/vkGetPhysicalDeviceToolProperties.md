# Usage Notes

Returns a list of active tools (validation layers, profilers,
debuggers, crash dump utilities) that are currently intercepting
Vulkan calls for this physical device.

Each `PhysicalDeviceToolProperties` includes the tool's name,
version, purposes (validation, profiling, tracing, etc.), and a
description.

Useful for diagnostics, log the active tools at startup to help
debug performance issues or unexpected validation messages. If no
tools are active, the list is empty.
