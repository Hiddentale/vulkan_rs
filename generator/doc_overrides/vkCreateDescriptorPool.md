# Usage Notes

Creates a pool from which descriptor sets are allocated. The pool
must be sized to accommodate all the descriptor sets and individual
descriptor types your application needs.

**Sizing**: specify `max_sets` (total descriptor sets) and a list of
`DescriptorPoolSize` entries that declare how many descriptors of
each type the pool holds. Under-sizing causes
`VK_ERROR_OUT_OF_POOL_MEMORY` at allocation time.

**Flags**:

- `FREE_DESCRIPTOR_SET`: allows individual sets to be freed with
  `free_descriptor_sets`. Without this flag, sets can only be
  reclaimed by resetting the entire pool.
- `UPDATE_AFTER_BIND`: required if any allocated set uses
  update-after-bind bindings.

**Common pattern**: create one pool per frame-in-flight. At the
start of each frame, `reset_descriptor_pool` reclaims all sets at
once, no individual tracking or freeing needed.
