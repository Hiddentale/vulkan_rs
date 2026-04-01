# Usage Notes

Unregisters a custom border color previously registered with
`register_custom_border_color_ext`, freeing the slot for reuse.

Ensure no samplers referencing this border color are in use
before unregistering.

Provided by `VK_EXT_descriptor_heap`.
