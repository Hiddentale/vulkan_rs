# Usage Notes

Queries the memory requirements for building a micromap. Returns
the destination micromap size and scratch memory size needed.

Provide a `MicromapBuildInfoEXT` with the triangle counts and
format. The addresses can be null, only the sizes and counts
matter for this query.

Use the results to allocate the micromap buffer and scratch buffer
before calling `cmd_build_micromaps_ext`.

Requires `VK_EXT_opacity_micromap`.
