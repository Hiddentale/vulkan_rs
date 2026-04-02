# Usage Notes

Vulkan 1.3 version of `cmd_copy_buffer` that uses an extensible
`CopyBufferInfo2` struct. Functionally identical to the 1.0 version
but supports future extensions via pNext.

Prefer this over `cmd_copy_buffer` when targeting Vulkan 1.3+.
