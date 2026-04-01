# Usage Notes

Checks whether a serialized acceleration structure (from
`copy_acceleration_structure_to_memory_khr`) is compatible with
this device and can be deserialized.

Returns `ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE` if the
data can be loaded, or `INCOMPATIBLE` if not. Incompatibility
typically means the data was serialized on different hardware or a
different driver version.

Check compatibility before attempting deserialization to avoid
errors.
