# Usage Notes

Returns the list of extensions supported by a physical device. Call
this to verify that required extensions are available before
requesting them in `DeviceCreateInfo::enabled_extension_names`.

Common extensions to check for:

- `VK_KHR_swapchain`: presentation to a surface (required for
  rendering to a window).
- `VK_KHR_dynamic_rendering`: render pass-less rendering (core in
  1.3).
- `VK_KHR_ray_tracing_pipeline`: hardware ray tracing.
- `VK_EXT_descriptor_indexing`: bindless descriptors (core in 1.2).

Pass `None` for `layer_name` to enumerate extensions provided by the
driver itself. Passing a layer name enumerates extensions provided by
that specific layer (rarely needed).

Requesting an unsupported extension at device creation causes
`VK_ERROR_EXTENSION_NOT_PRESENT`.
