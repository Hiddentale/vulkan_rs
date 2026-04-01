# Usage Notes

Queries the maximum workgroup size supported for subpass shading
on the given render pass. Returns an `Extent2D` with the max
width and height. Use this to configure subpass shading dispatch
parameters.

Requires `VK_HUAWEI_subpass_shading`.
