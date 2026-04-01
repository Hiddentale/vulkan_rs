# Usage Notes

Device-address variant of `cmd_copy_query_pool_results`. Copies
query results directly to a device address range instead of a
buffer handle.

`first_query` and `query_count` select which queries to copy.
`query_result_flags` controls formatting (`_64_BIT`, `WAIT`,
`WITH_AVAILABILITY`, `PARTIAL`).

The `StridedDeviceAddressRangeKHR` specifies the destination
address and stride between results.

Requires `VK_KHR_device_address_commands`.
