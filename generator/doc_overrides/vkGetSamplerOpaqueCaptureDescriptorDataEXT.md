# Usage Notes

Retrieves opaque capture data for a sampler descriptor. The
returned data can be used to reconstruct the descriptor in a
capture/replay scenario.

The sampler must have been created with
`CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT`.

Requires `VK_EXT_descriptor_buffer` and
`descriptorBufferCaptureReplay`.
