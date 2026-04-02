# Usage Notes

Acquires ownership of a swapchain image on Android. Takes a
native fence FD for synchronisation and can signal a Vulkan
semaphore or fence on completion. Android only.

Requires `VK_ANDROID_native_buffer`.
