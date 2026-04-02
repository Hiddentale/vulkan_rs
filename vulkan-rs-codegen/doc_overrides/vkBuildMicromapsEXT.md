# Usage Notes

Builds one or more micromaps on the host. This is the CPU-side
equivalent of `cmd_build_micromaps_ext`.

Each `MicromapBuildInfoEXT` specifies the source triangle data,
destination micromap, and scratch memory.

Requires `VK_EXT_opacity_micromap` and the
`micromapHostCommands` feature.
