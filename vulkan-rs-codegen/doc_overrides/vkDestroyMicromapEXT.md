# Usage Notes

Destroys a micromap created with `create_micromap_ext`. The
backing buffer is not freed, the application must manage buffer
lifetime separately.

Requires `VK_EXT_opacity_micromap`.
