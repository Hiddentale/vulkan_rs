# Usage Notes

Acquires the next available image from the swapchain for rendering.
Returns the index into the array from `get_swapchain_images_khr`.

**Synchronisation**: pass a semaphore, a fence, or both. The
semaphore/fence is signaled when the image is ready to be rendered
to. Do not start rendering until the semaphore is waited on in
`queue_submit`.

**Timeout**: in nanoseconds. `u64::MAX` blocks indefinitely.

**Special return values**:

- `VK_SUBOPTIMAL_KHR`: the swapchain still works but no longer
  matches the surface perfectly (e.g. after a resize). You can
  continue rendering but should recreate the swapchain soon.
- `VK_ERROR_OUT_OF_DATE_KHR`: the swapchain is incompatible with
  the surface and must be recreated before rendering. Do not present
  the acquired image.

A common frame loop:

```text
acquire_next_image_khr(swapchain, u64::MAX, image_available_sem, null)
// wait on image_available_sem in queue_submit
// render to swapchain_images[index]
// signal render_finished_sem in queue_submit
queue_present_khr(render_finished_sem, swapchain, index)
```
