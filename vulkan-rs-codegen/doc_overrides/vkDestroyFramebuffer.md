# Usage Notes

Destroys a framebuffer. No render pass instance using this
framebuffer may be pending execution.

Framebuffers are typically recreated whenever the swapchain is
resized, so they tend to have shorter lifetimes than most Vulkan
objects. With imageless framebuffers (Vulkan 1.2+) you can avoid
this churn entirely.
