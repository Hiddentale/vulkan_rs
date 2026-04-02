# Usage Notes

Vulkan 1.4 host-side image layout transition. Transitions an image
between layouts from the CPU without recording a command buffer.

The image must have been created with `HOST_TRANSFER` usage. The
transition happens synchronously on the calling thread.

This simplifies workflows where you need to transition an image
layout outside of a command buffer, for example, transitioning a
newly created host-transfer image from `UNDEFINED` to `GENERAL`
before performing host-side copies.

For GPU-side layout transitions (the common case), use
`cmd_pipeline_barrier2` with an image memory barrier.
