# Usage Notes

Vulkan 1.1 version of `get_physical_device_properties` that supports
chaining additional property structs via pNext.

Chain version-specific and extension property structs to query
extended limits and capabilities:

- `PhysicalDeviceVulkan11Properties`: subgroup properties, point
  clipping, protected memory.
- `PhysicalDeviceVulkan12Properties`: driver conformance, denorm
  behaviour, float controls, descriptor indexing limits, timeline
  semaphore properties.
- `PhysicalDeviceVulkan13Properties`: subgroup size control,
  inline uniform block limits, dynamic rendering limits.
- Extension structs like
  `PhysicalDeviceRayTracingPipelinePropertiesKHR`.

The base `PhysicalDeviceProperties` is identical to what
`get_physical_device_properties` returns.
