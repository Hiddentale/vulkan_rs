# Usage Notes

Returns the optional features supported by a physical device. Each
field is a boolean indicating whether the feature is available.

Query this before device creation to check whether features your
application needs are supported. Then enable only the features you
actually use in `DeviceCreateInfo::enabled_features`.

Commonly checked features:

- `sampler_anisotropy`: anisotropic texture filtering.
- `fill_mode_non_solid`: wireframe rendering.
- `wide_lines`: line widths other than 1.0.
- `geometry_shader`, `tessellation_shader`: optional shader stages.
- `multi_draw_indirect`: indirect draw with count > 1.
- `pipeline_statistics_query`: pipeline statistics queries.

Enabling a feature that is not supported causes device creation to
fail. Never blindly enable all features, only request what you need.

For extended features (Vulkan 1.1+), use
`get_physical_device_features2` with chained feature structs.
