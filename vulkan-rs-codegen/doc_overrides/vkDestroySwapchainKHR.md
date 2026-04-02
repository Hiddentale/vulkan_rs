# Usage Notes

Destroys a swapchain. All images obtained from
`get_swapchain_images_khr` become invalid, destroy any image views
and framebuffers referencing them first.

Wait for all rendering to complete (`device_wait_idle`) before
destroying. Do not destroy a swapchain while an acquired image has
not been presented.

When recreating a swapchain (e.g. on resize), create the new one
first (passing the old as `old_swapchain`), then destroy the old
one.

# Guide

See [Hello Triangle, Part 2](https://hiddentale.github.io/vulkan_rs/getting-started/hello-triangle-2.html) in the vulkan_rs guide.
