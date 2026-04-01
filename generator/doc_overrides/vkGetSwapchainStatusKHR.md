# Usage Notes

Queries the current status of a shared presentable swapchain
(created with `PRESENT_MODE_SHARED_DEMAND_REFRESH` or
`PRESENT_MODE_SHARED_CONTINUOUS_REFRESH`).

Returns `VK_SUCCESS` if the swapchain is usable, or
`VK_SUBOPTIMAL_KHR` / `VK_ERROR_OUT_OF_DATE_KHR` / surface-lost
errors if the swapchain needs recreation.

Only relevant for shared presentable images
(`VK_KHR_shared_presentable_image`). For regular swapchains, status
is communicated through `acquire_next_image_khr` and
`queue_present_khr` return values.
