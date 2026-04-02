# Usage Notes

Binds a sampler descriptor heap for use in subsequent draw and
dispatch commands. The `BindHeapInfoEXT` specifies the heap to
bind.

Sampler heaps hold sampler descriptors. Resource descriptors are
bound separately with `cmd_bind_resource_heap_ext`.

Provided by `VK_EXT_descriptor_heap`.
