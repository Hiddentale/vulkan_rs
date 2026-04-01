# Usage Notes

Copies an acceleration structure. Modes:

- `COPY_ACCELERATION_STRUCTURE_MODE_CLONE`: full copy. The
  destination must be at least as large as the source.
- `COPY_ACCELERATION_STRUCTURE_MODE_COMPACT`: copies into a smaller
  buffer after a compaction query. Use this to reclaim memory after
  building.

**Compaction workflow**:

1. Build with `BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION`.
2. Query compacted size with
   `cmd_write_acceleration_structures_properties_khr`.
3. Create a new, smaller backing buffer.
4. Copy with `MODE_COMPACT`.

Compaction typically saves 50–70% of BLAS memory.
