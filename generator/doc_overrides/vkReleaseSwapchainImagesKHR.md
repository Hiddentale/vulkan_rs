# Usage Notes

Releases acquired swapchain images back to the swapchain without
presenting them. Provided by `VK_KHR_swapchain_maintenance1`.

Use this when you have acquired an image but decided not to render
to it, for example, when aborting a frame due to a resize or
error. Without this extension, the only way to return an acquired
image is to present it.

The released images return to the pool and can be acquired again.
