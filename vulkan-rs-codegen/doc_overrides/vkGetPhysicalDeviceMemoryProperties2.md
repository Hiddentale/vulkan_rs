# Usage Notes

Vulkan 1.1 version of `get_physical_device_memory_properties` that
supports extensible output via pNext.

Chain `PhysicalDeviceMemoryBudgetPropertiesEXT` (if the
`VK_EXT_memory_budget` extension is available) to query per-heap
budget and current usage. This is essential for managing memory
pressure on systems with unified memory or limited VRAM.

The base `PhysicalDeviceMemoryProperties` (heaps and types) is
identical to what `get_physical_device_memory_properties` returns.
