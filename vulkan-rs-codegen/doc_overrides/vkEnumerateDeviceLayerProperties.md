# Usage Notes

Enumerates the available device-level validation layers. In modern
Vulkan (1.0.13+), device layers are deprecated in favour of
instance layers, this function exists for backwards compatibility
and typically returns the same list as instance layer enumeration.

Most applications do not need to call this. Enable validation layers
at instance creation via `InstanceCreateInfo::enabled_layer_names`
instead.
