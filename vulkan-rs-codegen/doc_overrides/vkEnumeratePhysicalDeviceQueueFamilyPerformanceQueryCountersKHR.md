# Usage Notes

Enumerates the performance counters available for a specific queue
family on a physical device. Returns `PerformanceCounterKHR`
structs (counter unit, storage type, UUID) and optionally fills
`PerformanceCounterDescriptionKHR` with human-readable names and
descriptions.

Use the counter indices when creating a performance query pool
with `QueryPoolPerformanceCreateInfoKHR`.

Requires `VK_KHR_performance_query`.
