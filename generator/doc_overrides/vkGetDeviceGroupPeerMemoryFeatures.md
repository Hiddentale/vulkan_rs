# Usage Notes

Queries the memory access capabilities between two physical devices
in a device group. Returns flags indicating whether memory allocated
on one device can be copied to, accessed generically, or accessed
natively from the other device.

Only relevant for multi-GPU device groups. On single-GPU systems
this is not needed.

Use the returned flags to decide how to share resources across
devices, for example, whether a texture on GPU 0 can be sampled
directly by GPU 1, or whether it must be copied.
