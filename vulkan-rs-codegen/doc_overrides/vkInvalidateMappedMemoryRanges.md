# Usage Notes

Invalidates CPU caches for mapped non-coherent memory so the CPU
can see data written by the GPU. The counterpart to
`flush_mapped_memory_ranges`.

Call this **after** the GPU has finished writing (e.g. after
`wait_for_fences` on the relevant submission) and **before** reading
the data through the mapped pointer.

The same alignment rules apply: `offset` and `size` must be
multiples of `non_coherent_atom_size`, or use offset zero with
`VK_WHOLE_SIZE`.

If you are using host-coherent memory, this call is unnecessary, 
GPU writes are automatically visible to the CPU. Most desktop GPUs
offer host-coherent memory types for host-visible heaps.
