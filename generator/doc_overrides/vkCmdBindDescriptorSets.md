# Usage Notes

Binds one or more descriptor sets to a command buffer at specified
set indices. Subsequent draw or dispatch calls read resources from
these bound sets.

**`first_set`**: the set index at which binding starts. Sets at
lower indices are not disturbed. This lets you bind per-frame data
at set 0 once and rebind only per-material or per-object sets at
higher indices.

**Dynamic offsets**: for descriptors of type
`UNIFORM_BUFFER_DYNAMIC` or `STORAGE_BUFFER_DYNAMIC`, the
`dynamic_offsets` slice provides byte offsets applied at bind time.
This lets multiple draw calls share a single large buffer with
different sub-regions without updating the descriptor set.

The bound pipeline layout and the descriptor set layouts must be
compatible, same binding layout at each set index. Binding a set
with an incompatible layout is undefined behaviour.
