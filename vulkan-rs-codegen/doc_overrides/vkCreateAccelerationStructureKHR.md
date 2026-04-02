# Usage Notes

Creates an acceleration structure for hardware ray tracing. An
acceleration structure is a spatial data structure (typically a BVH)
that the GPU traverses during ray intersection tests.

**Two levels**:

- **Bottom-level (BLAS)**: contains geometry (triangles or AABBs).
  Create one per mesh or mesh group.
- **Top-level (TLAS)**: contains instances that reference BLASes
  with per-instance transforms. Create one per scene.

The acceleration structure needs a backing buffer created with
`BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE`. Query the required
size with `get_acceleration_structure_build_sizes_khr` first.

After creation, build the structure with
`cmd_build_acceleration_structures_khr`. The structure is not usable
for tracing until built.
