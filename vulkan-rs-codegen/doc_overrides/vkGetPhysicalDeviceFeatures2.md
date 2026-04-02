# Usage Notes

Vulkan 1.1 version of `get_physical_device_features` that supports
chaining additional feature structs via pNext.

Chain version-specific and extension feature structs to query their
availability:

- `PhysicalDeviceVulkan11Features`
- `PhysicalDeviceVulkan12Features`
- `PhysicalDeviceVulkan13Features`
- Extension-specific structs like
  `PhysicalDeviceRayTracingPipelineFeaturesKHR`

Then pass the same chain (with desired features enabled) to
`DeviceCreateInfo` to enable them at device creation.

Always query before enabling, requesting an unsupported feature
fails device creation.
