# Usage Notes

Vulkan 1.1 version of `get_physical_device_queue_family_properties`
that supports extensible output via pNext.

Chain `QueueFamilyCheckpointPropertiesNV` for diagnostic checkpoint
support, or `QueueFamilyGlobalPriorityPropertiesKHR` for
global priority scheduling capabilities.

The base `QueueFamilyProperties` is identical to what
`get_physical_device_queue_family_properties` returns.
