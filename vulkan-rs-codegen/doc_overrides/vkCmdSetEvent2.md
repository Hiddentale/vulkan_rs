# Usage Notes

Vulkan 1.3 version of `cmd_set_event` that takes a `DependencyInfo`
describing the memory dependencies, rather than just a stage mask.

This provides more precise dependency information to the driver and
supports 64-bit stage and access flags. The dependency info specifies
exactly what memory accesses and pipeline stages are involved, which
can reduce unnecessary stalls.

Prefer this over `cmd_set_event` when targeting Vulkan 1.3+.
