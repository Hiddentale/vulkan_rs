# Usage Notes

Unmaps a previously mapped device memory region. After this call the
host pointer returned by `map_memory` is invalid.

Unmapping is only strictly necessary before `free_memory` if you
want to be explicit. Freeing memory implicitly unmaps it.

For persistently mapped memory (the recommended pattern), you
typically map once after allocation and unmap only during teardown.
There is no performance penalty for keeping memory mapped.

If the memory type is not `HOST_COHERENT`, make sure to call
`flush_mapped_memory_ranges` after your final writes before
unmapping, to ensure the GPU sees the latest data.
