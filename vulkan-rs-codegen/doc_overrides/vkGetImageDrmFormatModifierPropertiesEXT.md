# Usage Notes

Queries which DRM format modifier was selected for an image
created with `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`. The
chosen modifier determines the memory layout and is needed
when sharing the image with other DRM/KMS clients.

Requires `VK_EXT_image_drm_format_modifier`.
