# Usage Notes

Maps device memory into the host address space so the CPU can read or
write it. The memory must have been allocated from a memory type with
`MEMORY_PROPERTY_HOST_VISIBLE`.

**Persistently mapped memory**: it is valid (and recommended) to keep
memory mapped for the lifetime of the allocation. Map once after
`allocate_memory` and unmap only before `free_memory`. This avoids
repeated map/unmap overhead.

**Coherency**: if the memory type does not include
`MEMORY_PROPERTY_HOST_COHERENT`, you must call:

- `flush_mapped_memory_ranges` after CPU writes, before GPU reads.
- `invalidate_mapped_memory_ranges` after GPU writes, before CPU reads.

Host-coherent memory skips both calls at the cost of slightly lower
throughput on some architectures.

**Alignment**: when sub-allocating from a large allocation, ensure
offsets respect `non_coherent_atom_size` for flush/invalidate ranges.
