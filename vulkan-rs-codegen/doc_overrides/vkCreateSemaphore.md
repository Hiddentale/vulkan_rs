# Usage Notes

Creates a semaphore for GPU–GPU synchronization between queue
submissions. Unlike fences (CPU–GPU), semaphores are invisible to
the CPU, they are signaled and waited on entirely within
`queue_submit` or `queue_present_khr`.

**Binary semaphores** (the default) have two states: signaled and
unsignaled. A submission signals the semaphore, and a later
submission waits on it, which also resets it to unsignaled.

**Timeline semaphores** (Vulkan 1.2+) have a monotonically
increasing 64-bit counter. Create one by chaining
`SemaphoreTypeCreateInfo` with `SEMAPHORE_TYPE_TIMELINE`. Timeline
semaphores can be waited on and signaled from the CPU as well via
`wait_semaphores` and `signal_semaphore`.

Common uses:

- Synchronize between a graphics queue submit and a present.
- Order a transfer upload before a render pass that consumes it.
- Coordinate work across different queue families.

# Guide

See [Synchronization](https://hiddentale.github.io/vulkan_rs/concepts/synchronization.html) in the vulkan_rs guide.
