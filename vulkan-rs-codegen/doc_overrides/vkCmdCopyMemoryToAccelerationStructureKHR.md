# Usage Notes

Deserializes an acceleration structure from a buffer that was
previously written by
`cmd_copy_acceleration_structure_to_memory_khr`.

The data must have been serialized on the same driver and hardware
(check `acceleration_structure_uuid` compatibility before loading).

After deserialization the acceleration structure is ready for use
in ray tracing commands.
