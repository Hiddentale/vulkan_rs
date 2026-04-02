# Usage Notes

Returns general properties of a physical device: device name, vendor
ID, driver version, API version, device type, and device limits.

Key fields to check during device selection:

- **`device_type`**: `DISCRETE_GPU`, `INTEGRATED_GPU`, `VIRTUAL_GPU`,
  `CPU`, or `OTHER`. Discrete GPUs typically offer the best
  performance.
- **`api_version`**: the highest Vulkan version the device supports.
- **`limits`**: contains hundreds of device limits like
  `max_image_dimension_2d`, `max_push_constants_size`,
  `timestamp_period`, `non_coherent_atom_size`, etc.
- **`pipeline_cache_uuid`**: used to validate pipeline cache data
  across sessions.

For extended properties (Vulkan 1.1+), use
`get_physical_device_properties2` which supports chaining additional
property structs like `PhysicalDeviceVulkan12Properties`.

# Guide

See [Hello Triangle, Part 1](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html) in the vulkan_rs guide.
