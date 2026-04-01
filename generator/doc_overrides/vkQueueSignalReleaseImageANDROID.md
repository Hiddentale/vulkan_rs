# Usage Notes

Releases a swapchain image back to the Android compositor after
rendering. Waits on the given semaphores and returns a native
fence FD for external synchronisation. Android only.

Requires `VK_ANDROID_native_buffer`.
