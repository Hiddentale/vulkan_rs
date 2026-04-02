# Usage Notes

Returns the array of presentable images owned by the swapchain. You
do not create or destroy these images, they are managed by the
swapchain.

The returned image count may be greater than `min_image_count`
requested at swapchain creation.

Create an `ImageView` for each swapchain image to use them as
render targets. These views (and any framebuffers using them) must
be destroyed before the swapchain is destroyed.

The images start in an undefined layout. Transition them to the
appropriate layout (e.g. `COLOR_ATTACHMENT_OPTIMAL`) during the
first render pass or via a pipeline barrier.
