# Usage Notes

Creates a Fuchsia buffer collection that negotiates memory
constraints between Vulkan and other Fuchsia services (e.g.
scenic, camera). Fuchsia OS only. After creation, set buffer
or image constraints before allocating.

Requires `VK_FUCHSIA_buffer_collection`.
