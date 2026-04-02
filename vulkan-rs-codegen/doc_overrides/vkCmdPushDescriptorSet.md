# Usage Notes

Pushes descriptor updates directly into the command buffer without
allocating a descriptor set from a pool. The descriptors are embedded
in the command stream and only live for the duration of the current
command buffer recording.

**Advantages**:

- No descriptor pool allocation or management.
- No need to track descriptor set lifetimes.
- Ideal for per-draw data that changes every frame.

**Trade-offs**:

- Inflates command buffer size (descriptors are stored inline).
- Not suitable for large descriptor sets, use conventional
  allocated sets for sets with many bindings.

The pipeline layout must have been created with
`DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR` on the target set
index.

Core in Vulkan 1.4. Previously available via `VK_KHR_push_descriptor`.
