# Usage Notes

Queries the sparse memory requirements for an image created with
one of the `IMAGE_CREATE_SPARSE_*` flags. Returns a list of sparse
image format properties describing the memory layout for each
image aspect (color, depth, stencil, metadata).

Sparse resources allow partially-resident textures where only some
mip levels or regions are backed by physical memory. This is an
advanced feature primarily used for virtual texturing and terrain
streaming.

If the image was not created with sparse flags, this returns an
empty list. Check `physical_device_features.sparse_binding` before
using sparse resources.
