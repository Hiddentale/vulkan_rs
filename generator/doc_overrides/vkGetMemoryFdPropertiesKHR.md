# Usage Notes

Queries which memory types are compatible with an external file
descriptor. Use this when importing memory from an fd received
from another process or API.

Returns `MemoryFdPropertiesKHR` with a `memory_type_bits` bitmask
indicating which memory type indices can be used when allocating
memory to import this fd.

Only valid for `DMA_BUF` handle types. `OPAQUE_FD` handles don't
need this query, their memory type is determined by the
exporting allocation.
