# Usage Notes

Returns a 64-bit GPU virtual address for a buffer. This address can
be passed to shaders via push constants or descriptors, enabling
direct pointer-style access to buffer data from GPU code.

The buffer must have been created with
`BUFFER_USAGE_SHADER_DEVICE_ADDRESS` and the
`buffer_device_address` feature must be enabled.

**Use cases**:

- **Bindless rendering**: pass buffer addresses in a storage buffer
  or push constant instead of binding individual descriptors.
- **Acceleration structures**: ray tracing BLASes and TLASes
  reference geometry buffers by device address.
- **GPU-driven pipelines**: indirect command generators read vertex
  and index data by address.

The address remains valid for the lifetime of the buffer. If the
buffer was created with
`BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY`, the address can be
captured and replayed across sessions.
