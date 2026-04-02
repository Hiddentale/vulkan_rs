# Usage Notes

Returns the list of queue families supported by a physical device.
Each queue family has a set of capabilities (graphics, compute,
transfer, sparse binding) and a count of available queues.

You must query this before creating a device to determine which
queue family indices to request.

**Common queue family selection**:

- **Graphics + compute**: most desktop GPUs have a single family
  that supports both. Use it for all rendering and compute work.
- **Dedicated transfer**: some GPUs expose a transfer-only family
  backed by a DMA engine. Use it for async uploads to overlap with
  rendering.
- **Dedicated compute**: some GPUs expose a compute-only family
  for async compute.

Check `queue_flags` for `QUEUE_GRAPHICS`, `QUEUE_COMPUTE`,
`QUEUE_TRANSFER`, and `QUEUE_SPARSE_BINDING`. Also check
`timestamp_valid_bits` if you need GPU timestamps on that queue.

For extended properties (Vulkan 1.1+), use
`get_physical_device_queue_family_properties2`.
