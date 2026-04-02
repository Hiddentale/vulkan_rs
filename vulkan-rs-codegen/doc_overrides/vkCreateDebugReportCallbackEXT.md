# Usage Notes

Creates a legacy debug report callback. Superseded by
`create_debug_utils_messenger_ext` (`VK_EXT_debug_utils`),
which provides richer message data and object labeling.

The callback receives validation messages filtered by the
`flags` bitmask (error, warning, performance, info, debug).

Destroy with `destroy_debug_report_callback_ext`.

Requires `VK_EXT_debug_report`. Prefer `VK_EXT_debug_utils`
for new code.
