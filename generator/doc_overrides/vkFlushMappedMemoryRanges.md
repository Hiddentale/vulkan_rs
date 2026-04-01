# Usage Notes

Flushes CPU writes to mapped non-coherent memory so the GPU can see
them. Only needed when the memory type does **not** include
`MEMORY_PROPERTY_HOST_COHERENT`.

Call this **after** writing data through the mapped pointer and
**before** the GPU reads it (i.e. before the relevant
`queue_submit`).

**Alignment**: the `offset` and `size` of each range must be
multiples of `non_coherent_atom_size` (from physical device limits),
or `offset` must be zero and `size` must be `VK_WHOLE_SIZE`. Failing
to align causes undefined behaviour on some implementations.

Multiple ranges can be flushed in a single call. Batch them when
updating several sub-allocations within the same memory object.

If you are using host-coherent memory, this call is unnecessary and
can be skipped entirely.
