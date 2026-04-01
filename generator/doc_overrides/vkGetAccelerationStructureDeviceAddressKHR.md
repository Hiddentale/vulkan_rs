# Usage Notes

Returns the GPU device address of an acceleration structure. This
address is used when building a TLAS, each instance in the TLAS
references a BLAS by its device address.

The address remains valid for the lifetime of the acceleration
structure.
