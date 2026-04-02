# Usage Notes

Returns a list of all physical devices (GPUs) available to the
instance. This is typically the first call after instance creation, 
you need a physical device to query capabilities and create a
logical device.

The order of physical devices is driver-dependent and not guaranteed
to be stable across runs. To pick the right GPU:

1. Enumerate all devices.
2. Query `get_physical_device_properties` for each.
3. Prefer `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` for performance, or
   `INTEGRATED_GPU` for power efficiency.
4. Check queue families, memory heaps, and required features.

On systems with multiple GPUs (e.g. a discrete + integrated), this
returns all of them. Vulkan does not have a concept of a "default"
GPU, your application must choose.

# Guide

See [Hello Triangle, Part 1](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-1.html) in the vulkan_rs guide.
