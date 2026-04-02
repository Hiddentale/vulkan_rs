# Usage Notes

Vulkan 1.1 version of `get_physical_device_image_format_properties`
that supports extensible input and output via pNext.

Chain in the input:

- `PhysicalDeviceExternalImageFormatInfo`: query external memory
  compatibility for the image.
- `PhysicalDeviceImageDrmFormatModifierInfoEXT`: query DRM modifier
  support.

Chain in the output:

- `ExternalImageFormatProperties`: external memory capabilities.
- `SamplerYcbcrConversionImageFormatProperties`: YCBCR support.

Returns `VK_ERROR_FORMAT_NOT_SUPPORTED` if the combination is not
supported.
