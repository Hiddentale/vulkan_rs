# Usage Notes

Queries the present capabilities of a device group, which physical
devices can present to which surfaces, and what presentation modes
are supported.

Only relevant for multi-GPU device groups. On single-GPU systems,
only `DEVICE_GROUP_PRESENT_MODE_LOCAL` is supported (each device
presents its own images).
