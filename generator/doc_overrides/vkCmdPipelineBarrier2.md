# Usage Notes

Vulkan 1.3 version of `cmd_pipeline_barrier` that uses
`DependencyInfo` with `MemoryBarrier2`, `BufferMemoryBarrier2`, and
`ImageMemoryBarrier2` structs.

The key improvement over the 1.0 version is that stage and access
masks are specified **per barrier** rather than globally for the
entire call. This gives the driver more precise dependency
information, which can reduce unnecessary stalls.

The 1.3 barrier structs also use 64-bit stage and access flags,
supporting stages and access types that do not fit in the original
32-bit fields (e.g. ray tracing, mesh shading).

Prefer this over `cmd_pipeline_barrier` when targeting Vulkan 1.3+.
