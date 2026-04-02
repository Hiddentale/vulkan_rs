# Usage Notes

Host-side query of micromap properties. This is the CPU equivalent
of `cmd_write_micromaps_properties_ext`.

Typically used to query compacted size
(`QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT`) without going through
a query pool.

Requires `VK_EXT_opacity_micromap` and the
`micromapHostCommands` feature.
