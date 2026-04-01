# Usage Notes

Binds a resource descriptor heap for use in subsequent draw and
dispatch commands. The `BindHeapInfoEXT` specifies the heap to
bind.

Resource heaps hold descriptors for buffers, images, and
acceleration structures. Samplers are bound separately with
`cmd_bind_sampler_heap_ext`.

Provided by `VK_EXT_descriptor_heap`.
