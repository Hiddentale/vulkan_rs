# Usage Notes

Manually injects a message into the legacy debug report callback
chain. All registered callbacks matching the specified `flags`
will receive the message.

`p_layer_prefix` and `p_message` are C strings. `object` is the
raw handle of the relevant Vulkan object (or 0 if none).

Superseded by `submit_debug_utils_message_ext`.

Requires `VK_EXT_debug_report`.
