# Usage Notes

Retrieves a host pointer to the internal memory backing a
descriptor set. Allows direct CPU writes to descriptor data,
bypassing the normal `update_descriptor_sets` path for lower
overhead. The layout must match what was queried with
`get_descriptor_set_layout_host_mapping_info_valve`.

Requires `VK_VALVE_descriptor_set_host_mapping`.
