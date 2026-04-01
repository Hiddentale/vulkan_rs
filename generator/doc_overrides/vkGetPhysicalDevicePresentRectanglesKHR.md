# Usage Notes

Returns the set of rectangular regions that cover the presentable
area of a surface for a device group. Each rectangle represents a
region that one physical device in the group is responsible for
presenting.

Only relevant for multi-GPU device groups with
`DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE`. On single-GPU
systems, this returns a single rectangle covering the entire
surface.
