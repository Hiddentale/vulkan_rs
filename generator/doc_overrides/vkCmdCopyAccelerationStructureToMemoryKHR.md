# Usage Notes

Serializes an acceleration structure into a buffer for storage or
transfer. The serialized format is opaque and driver-specific, it
can only be deserialized on the same driver and hardware.

Use this for:

- Saving acceleration structures to disk for faster subsequent loads.
- Transferring structures between devices in a device group.

Deserialize with `cmd_copy_memory_to_acceleration_structure_khr`.
