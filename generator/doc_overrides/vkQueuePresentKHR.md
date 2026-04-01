# Usage Notes

Presents a rendered swapchain image to the display. This is the
final step in the frame loop, after rendering is complete, present
the image to make it visible.

**Wait semaphores**: the present waits on these semaphores before
presenting. Pass the semaphore that your render submission signals
to ensure the image is fully rendered before it goes to the display.

**Multiple swapchains**: a single present call can present to
multiple swapchains simultaneously (e.g. for multi-window or
multi-monitor rendering).

**Return values**:

- `VK_SUBOPTIMAL_KHR`: presented successfully but the swapchain
  should be recreated.
- `VK_ERROR_OUT_OF_DATE_KHR`: presentation failed, the swapchain
  must be recreated.

The present queue does not need to be the same as the graphics
queue, but the semaphore synchronisation must be correct if they
differ.
