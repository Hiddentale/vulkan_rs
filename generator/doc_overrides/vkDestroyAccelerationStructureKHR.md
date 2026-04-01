# Usage Notes

Destroys an acceleration structure. The structure must not be
referenced by any pending ray tracing command or TLAS build.

Destroying the acceleration structure does not free the backing
buffer, destroy or reclaim it separately.
