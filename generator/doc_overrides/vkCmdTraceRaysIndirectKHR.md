# Usage Notes

Dispatches rays with launch dimensions read from a GPU buffer.
Identical to `cmd_trace_rays_khr` except the `width`, `height`,
and `depth` are sourced from a `TraceRaysIndirectCommandKHR`
struct at `indirect_device_address`.

This enables the GPU to determine ray dispatch dimensions without
a CPU round-trip, useful when the dispatch size depends on prior
GPU work such as culling, tile classification, or adaptive
sampling.

The indirect buffer must have been created with
`BUFFER_USAGE_INDIRECT_BUFFER` and the address must be aligned
to 4 bytes. The SBT parameters are still provided directly on
the CPU side.

Requires the `rayTracingPipelineTraceRaysIndirect` feature.
