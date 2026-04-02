# Usage Notes

Binds sparse memory regions to sparse resources (buffers or images).
This is the only way to change the memory backing of a sparse
resource after creation.

Sparse binding supports:

- **Partial residency**: bind memory to individual mip tail regions
  or image tiles, leaving others unbound.
- **Aliasing**: multiple sparse resources can alias the same memory
  region (with `IMAGE_CREATE_SPARSE_ALIASED`).
- **Dynamic re-binding**: swap memory pages at runtime for virtual
  texturing or streaming.

The bind operation is asynchronous and can synchronize with
semaphores, similar to `queue_submit`. The queue must support sparse
binding (check `QUEUE_SPARSE_BINDING`).

This is an advanced feature. Most applications use fully-bound
resources with `bind_buffer_memory` / `bind_image_memory` instead.
