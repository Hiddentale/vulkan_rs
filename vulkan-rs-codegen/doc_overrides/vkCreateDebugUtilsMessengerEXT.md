# Usage Notes

Creates a debug messenger that receives validation layer messages,
performance warnings, and general debug info via a user-provided
callback.

`DebugUtilsMessengerCreateInfoEXT` configures:
- `message_severity`: which severities to receive (verbose, info,
  warning, error).
- `message_type`: which categories (general, validation,
  performance).
- `pfn_user_callback`: your callback function.

Create the messenger immediately after the instance for maximum
coverage. Destroy with `destroy_debug_utils_messenger_ext`.

Requires `VK_EXT_debug_utils`. Supersedes `VK_EXT_debug_report`.
