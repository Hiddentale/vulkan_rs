# Usage Notes

Writes or copies resource bindings into descriptor sets. This is
how you connect actual buffers, images, and samplers to the
descriptor slots that shaders read from.

**Writes** (`WriteDescriptorSet`): bind concrete resources to a
specific set + binding + array element. Each write targets one
descriptor type (uniform buffer, combined image sampler, storage
buffer, etc.).

**Copies** (`CopyDescriptorSet`): duplicate bindings from one set
to another. Rarely used, writes cover nearly all cases.

Updates take effect immediately and are visible to any command buffer
recorded after the update. However, updating a set that is currently
bound in a pending command buffer is undefined behaviour unless the
set was allocated from a pool with `UPDATE_AFTER_BIND` and the
binding is marked as update-after-bind in the layout.

Batch multiple writes into a single call when possible, the driver
can often process them more efficiently.
