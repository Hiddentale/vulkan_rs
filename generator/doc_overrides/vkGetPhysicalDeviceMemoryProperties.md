# Usage Notes

Returns the memory heaps and memory types available on the physical
device. Essential for choosing the right memory type when allocating
device memory.

**Memory heaps**: represent physical memory pools (e.g. VRAM, system
RAM). Each heap has a size and flags (`DEVICE_LOCAL` for GPU memory).

**Memory types**: each type references a heap and has property flags:

- `DEVICE_LOCAL`: fast GPU access. Preferred for images, vertex
  buffers, and anything the GPU reads frequently.
- `HOST_VISIBLE`: can be mapped for CPU access. Required for staging
  buffers and any CPU-written data.
- `HOST_COHERENT`: mapped writes are automatically visible to the
  GPU without explicit flushes.
- `HOST_CACHED`: CPU reads are fast (cached). Useful for readback
  buffers.
- `LAZILY_ALLOCATED`: memory may not be backed until used. For
  transient attachments on tile-based GPUs.

**Choosing a memory type**: AND the `memory_type_bits` from
`get_buffer_memory_requirements` or `get_image_memory_requirements`
with your desired property flags. Pick the first matching type.

For extended properties (Vulkan 1.1+), use
`get_physical_device_memory_properties2`.
