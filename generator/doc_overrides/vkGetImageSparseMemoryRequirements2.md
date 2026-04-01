# Usage Notes

Vulkan 1.1 version of `get_image_sparse_memory_requirements` that
supports extensible output structs via pNext. Returns the sparse
memory requirements for an image created with sparse flags.

For non-sparse images, returns an empty list. Only relevant if you
are using sparse resources.
