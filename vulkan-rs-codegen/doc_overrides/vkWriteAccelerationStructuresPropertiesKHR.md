# Usage Notes

Host-side query of acceleration structure properties. The CPU
counterpart to `cmd_write_acceleration_structures_properties_khr`.

Writes results directly to a host buffer rather than a query pool.
Supports the same query types: compacted size and serialization
size.

Requires the `acceleration_structure_host_commands` feature.
