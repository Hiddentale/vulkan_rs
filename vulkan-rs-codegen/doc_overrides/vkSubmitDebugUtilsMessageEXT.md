# Usage Notes

Manually injects a debug message into the debug utils callback
chain. The message is delivered to all active debug messengers
that match the specified severity and type flags.

Useful for application-level diagnostics, e.g., logging a
warning when a resource limit is approached.

The `DebugUtilsMessengerCallbackDataEXT` carries the message
string, message ID, and optional object labels/queue labels for
context.

Requires `VK_EXT_debug_utils`.
