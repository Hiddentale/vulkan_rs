# Usage Notes

Vulkan 1.1 version of `get_physical_device_format_properties` that
supports extensible output via pNext.

Chain `DrmFormatModifierPropertiesListEXT` to query DRM format
modifier support, or `FormatProperties3` (Vulkan 1.3) for extended
format feature flags that do not fit in the original 32-bit fields.

The base `FormatProperties` (linear, optimal, buffer features) is
identical to what `get_physical_device_format_properties` returns.
