# Usage Notes

Records a GPU-side acceleration structure build or update. This is
the primary way to build BLASes and TLASes for ray tracing.

**Build vs update**: an initial build creates the structure from
scratch. An update (`BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE`)
modifies an existing structure in-place, which is faster but
produces lower traversal quality. Use updates for dynamic geometry
(e.g. animated characters) and full rebuilds when geometry changes
significantly.

**Scratch buffer**: builds require a temporary scratch buffer.
Query the required size with
`get_acceleration_structure_build_sizes_khr` and create a buffer
with `BUFFER_USAGE_STORAGE_BUFFER`.

Multiple builds can be batched in a single call. The driver may
execute them in parallel.

Must be recorded outside a render pass.
