# Usage Notes

Queries the host memory layout for a specific binding within a
descriptor set layout. Returns the stride and offset needed to
write descriptors directly via the host pointer obtained from
`get_descriptor_set_host_mapping_valve`.

Requires `VK_VALVE_descriptor_set_host_mapping`.
