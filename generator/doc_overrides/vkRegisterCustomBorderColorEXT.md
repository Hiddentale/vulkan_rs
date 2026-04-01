# Usage Notes

Registers a custom border color for use with samplers. Returns a
handle that can be referenced when creating samplers with
`BORDER_COLOR_CUSTOM` modes.

The device has a limited number of custom border color slots
(query `maxCustomBorderColors`).

Unregister with `unregister_custom_border_color_ext` when no
longer needed.

Provided by `VK_EXT_descriptor_heap`.
