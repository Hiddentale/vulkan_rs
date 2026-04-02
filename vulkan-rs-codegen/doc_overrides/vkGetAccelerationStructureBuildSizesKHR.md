# Usage Notes

Queries the buffer sizes needed to build an acceleration structure:
the final structure size, the scratch buffer size for builds, and
the scratch buffer size for updates.

Call this before creating the acceleration structure and scratch
buffers. The `max_primitive_counts` parameter specifies the maximum
number of primitives per geometry, the returned sizes are
worst-case guarantees for those counts.

The actual built size may be smaller. For BLASes, build with
`ALLOW_COMPACTION` and query the compacted size afterwards to
reclaim excess memory.
