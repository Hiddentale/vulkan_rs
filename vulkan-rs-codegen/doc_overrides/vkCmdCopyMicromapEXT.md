# Usage Notes

Copies or compacts a micromap. The `CopyMicromapInfoEXT` specifies
source, destination, and mode (`CLONE` or `COMPACT`).

Use `COMPACT` after building with `BUILD_ALLOW_COMPACTION` to
reduce memory usage. Query the compacted size with
`cmd_write_micromaps_properties_ext`.

Requires `VK_EXT_opacity_micromap`.
