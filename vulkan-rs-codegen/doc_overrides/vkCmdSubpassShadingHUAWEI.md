# Usage Notes

Dispatches subpass shading work within the current subpass.
Subpass shading runs compute-like shaders that have access to
input attachments at the fragment's location, combining the
efficiency of compute with the data locality of subpasses.

Requires `VK_HUAWEI_subpass_shading`.
