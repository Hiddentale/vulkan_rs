# Usage Notes

Binds memory to one or more images in a single call. This is the
Vulkan 1.1 batch version of `bind_image_memory`.

Also required when binding memory to images created with
disjoint multi-planar formats, each plane is bound separately via
`BindImagePlaneMemoryInfo` in the pNext chain.

For device groups (multi-GPU), chain
`BindImageMemoryDeviceGroupInfo` to assign memory per device and
specify split-instance bind regions.
