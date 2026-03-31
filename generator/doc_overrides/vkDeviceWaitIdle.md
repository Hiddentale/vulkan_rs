# Usage Notes

Blocks the calling thread until **all** queues on this device are
idle. This is the nuclear option for synchronization — it drains
every queue completely.

Typical uses:

- **Before destroying the device**: ensures no GPU work is in flight
  before you start tearing down resources.
- **Before a swapchain resize**: guarantees all frames are done so
  image views and framebuffers can be safely recreated.

Avoid calling this in a render loop. It forces a full CPU–GPU
round-trip and prevents any overlap. Use per-frame fences or
timeline semaphores instead.
