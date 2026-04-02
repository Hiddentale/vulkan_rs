# Usage Notes

Dynamically updates the priority of a device memory allocation.
Higher-priority allocations are less likely to be evicted under
memory pressure. Use this to promote frequently accessed
resources or demote resources that are no longer critical.

Requires `VK_EXT_pageable_device_local_memory`.
