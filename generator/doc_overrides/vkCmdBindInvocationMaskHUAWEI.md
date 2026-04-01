# Usage Notes

Binds an image view as an invocation mask for ray tracing. The
mask selectively enables or disables ray generation shader
invocations per pixel, allowing efficient partial-screen ray
tracing on Huawei GPUs.

Requires `VK_HUAWEI_invocation_mask`.
