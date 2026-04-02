# Usage Notes

Vulkan 1.3 command that queries sparse memory requirements for an
image **without creating it first**. The counterpart to
`get_device_image_memory_requirements` for sparse images.

Only relevant if you are using sparse resources with the hypothetical
image creation parameters.
