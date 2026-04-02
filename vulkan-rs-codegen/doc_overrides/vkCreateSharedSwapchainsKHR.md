# Usage Notes

Creates one or more swapchains that share presentable images with
their `old_swapchain`. Provided by `VK_KHR_display_swapchain`.

This is primarily used for direct-to-display rendering where
multiple swapchains share the same display plane. For window-based
rendering, use `create_swapchain_khr` instead.
