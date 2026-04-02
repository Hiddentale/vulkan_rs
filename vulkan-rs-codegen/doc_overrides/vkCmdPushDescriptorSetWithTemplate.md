# Usage Notes

Combines push descriptors with descriptor update templates for
maximum efficiency. Updates are pushed directly into the command
buffer using the compact host data format defined by the template.

This is the fastest path for per-draw descriptor updates: no pool
allocation, no `WriteDescriptorSet` array construction, and the
driver has a pre-compiled update path from the template.

The template must have been created with
`DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS`.

Core in Vulkan 1.4. Previously via `VK_KHR_push_descriptor` +
`VK_KHR_descriptor_update_template`.
