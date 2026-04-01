# Usage Notes

Writes acceleration structure properties into a query pool. The
primary use is querying `QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE`
after a build to determine the compacted size before copying.

**Compaction workflow**:

1. Build with `ALLOW_COMPACTION`.
2. `cmd_write_acceleration_structures_properties_khr` with
   `COMPACTED_SIZE` query type.
3. Read the result from the query pool.
4. Create a smaller buffer and copy with `MODE_COMPACT`.

Also supports `QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE`
for estimating serialization buffer requirements.
