# Usage Notes

Begins a custom resolve region, allowing the application to use
its own fragment shader for MSAA resolve instead of the fixed-
function resolve. End with `cmd_end_custom_resolve_ext`.

Useful for tone-mapped or weighted resolves that the built-in
resolve operations cannot express.

Requires `VK_EXT_custom_resolve`.
