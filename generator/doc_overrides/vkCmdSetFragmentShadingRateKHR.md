# Usage Notes

Sets the pipeline fragment shading rate and combiner operations
as dynamic state. This controls how many pixels each fragment
shader invocation covers, larger fragment sizes reduce shading
cost at the expense of detail.

`p_fragment_size` specifies the fragment size (e.g., 1x1, 1x2,
2x2, 2x4, 4x4). Not all sizes are supported, query
`get_physical_device_fragment_shading_rates_khr` for the list.

`combiner_ops` defines how the pipeline rate, primitive rate
(from the vertex shader), and attachment rate (from a shading
rate image) are combined to produce the final rate.

Requires `VK_KHR_fragment_shading_rate` and the
`pipelineFragmentShadingRate` device feature.
