# Usage Notes

Builds or updates acceleration structures on the **host** (CPU).
This is the CPU-side alternative to
`cmd_build_acceleration_structures_khr`.

Host builds are useful for offline processing, tools, or when GPU
build capacity is limited. However, GPU builds are significantly
faster for real-time applications.

Requires the `acceleration_structure_host_commands` feature.
