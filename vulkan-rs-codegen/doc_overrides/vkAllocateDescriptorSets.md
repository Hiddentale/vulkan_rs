# Usage Notes

Allocates descriptor sets from a descriptor pool. Each set is backed
by one of the `set_layouts` in the allocate info.

Common failure causes:

- **Pool exhaustion**: the pool does not have enough descriptors of
  the required types, or the maximum set count has been reached.
  Returns `VK_ERROR_OUT_OF_POOL_MEMORY`. Pre-calculate pool sizes
  carefully or use `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`
  to allow freeing and reallocation.
- **Fragmentation**: even with enough total descriptors, internal
  fragmentation can cause allocation failure. Resetting the entire
  pool with `reset_descriptor_pool` defragments it.

Descriptor sets become invalid when their parent pool is destroyed
or reset. Do not submit command buffers that reference descriptor
sets from a pool that has been reset.

For frequently-updated descriptors, consider
`VK_KHR_push_descriptor` which avoids set allocation entirely.
