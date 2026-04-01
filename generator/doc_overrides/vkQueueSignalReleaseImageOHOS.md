# Usage Notes

Releases a swapchain image back to the OHOS compositor after
rendering. Waits on the given semaphores and returns a native
fence FD for external synchronisation. OHOS only.

Requires `VK_OHOS_native_buffer`.
