# Usage Notes

Destroys a Vulkan instance and frees all instance-level resources.
All devices created from this instance must be destroyed first.

Safe teardown order:

1. Destroy all `Device` objects (which destroys all device-child
   objects).
2. Destroy any debug messengers or debug report callbacks.
3. Destroy any surfaces.
4. `destroy_instance`.

After this call the instance handle and all physical device handles
obtained from it are invalid.

# Guide

See [The Vulkan Object Model](https://hiddentale.github.io/vulkan_rs/concepts/object-model.html) in the vulkan_rs guide.
