# Usage Notes

Binds embedded immutable samplers from a descriptor set layout
that was created with `CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT`.
These samplers are baked into the layout and do not need buffer
memory.

Specify the `pipeline_bind_point`, `layout`, and `set` index.

For the pNext-extensible variant, see
`cmd_bind_descriptor_buffer_embedded_samplers2_ext`.

Requires `VK_EXT_descriptor_buffer`.
