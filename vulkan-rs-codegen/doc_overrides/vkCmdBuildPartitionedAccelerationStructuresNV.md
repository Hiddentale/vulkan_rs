# Usage Notes

Builds a partitioned acceleration structure where instances are
grouped into independently updatable partitions. This allows
updating subsets of the TLAS without rebuilding the entire
structure.

Requires `VK_NV_partitioned_acceleration_structure`.
