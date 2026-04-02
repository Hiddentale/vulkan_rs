# Usage Notes

Vulkan 1.3 version of `cmd_wait_events` that takes per-event
`DependencyInfo` structs instead of global stage masks and barriers.

Each event can have its own set of memory barriers and stage masks,
giving the driver more precise information about what each event
protects.

Prefer this over `cmd_wait_events` when targeting Vulkan 1.3+.
