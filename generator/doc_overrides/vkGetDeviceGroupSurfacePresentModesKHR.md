# Usage Notes

Queries which device group present modes a surface supports. The
returned bitmask indicates whether `LOCAL`, `REMOTE`, `SUM`, or
`LOCAL_MULTI_DEVICE` modes are available.

Only relevant for multi-GPU device groups. On single-GPU systems,
only `LOCAL` is supported.
