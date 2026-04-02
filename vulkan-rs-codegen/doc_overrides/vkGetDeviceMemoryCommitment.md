# Usage Notes

Queries how many bytes of a lazily-allocated memory object are
currently backed by physical storage. Only meaningful for memory
allocated with `MEMORY_PROPERTY_LAZILY_ALLOCATED`.

Lazily-allocated memory is primarily used for transient framebuffer
attachments on tile-based GPUs (mobile). The driver may not back the
full allocation with physical memory until tiles actually need it.

On desktop GPUs this typically returns the full allocation size since
lazy allocation is rarely supported. Check
`memory_properties.memory_types` for the `LAZILY_ALLOCATED` flag
before relying on this.
