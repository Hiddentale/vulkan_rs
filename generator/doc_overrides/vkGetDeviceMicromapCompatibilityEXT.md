# Usage Notes

Checks whether a serialized micromap is compatible with the
current device. Returns `COMPATIBLE` or `INCOMPATIBLE`.

Use this before deserializing a micromap (via
`cmd_copy_memory_to_micromap_ext` or `copy_memory_to_micromap_ext`)
to verify it will work on this device.

Requires `VK_EXT_opacity_micromap`.
