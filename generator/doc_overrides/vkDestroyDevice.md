# Usage Notes

Destroys a logical device and frees its resources. All objects
created from this device (buffers, images, pipelines, command pools,
etc.) **must** be destroyed before calling this.

A safe teardown order:

1. `device_wait_idle` — ensure no GPU work is in flight.
2. Destroy all device-child objects (pipelines, buffers, images,
   views, descriptor pools, command pools, fences, semaphores, etc.).
3. `free_memory` for all device memory allocations.
4. `destroy_device`.

After this call the `Device` handle is invalid. Do not use it or any
object created from it.
