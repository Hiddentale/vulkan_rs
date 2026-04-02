# Usage Notes

Sets the compute occupancy priority for subsequent dispatch
commands. Higher priority may increase the number of warps
resident on an SM, trading off per-warp resources for greater
parallelism.

Requires `VK_NV_compute_occupancy_priority`.
