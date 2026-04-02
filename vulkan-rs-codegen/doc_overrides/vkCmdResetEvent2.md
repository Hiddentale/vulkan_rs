# Usage Notes

Vulkan 1.3 version of `cmd_reset_event` that uses 64-bit pipeline
stage flags. Supports newer stages not available in the original
32-bit field.

Prefer this over `cmd_reset_event` when targeting Vulkan 1.3+.
