# Usage Notes

Retrieves the `u64` value previously stored on a Vulkan object with
`set_private_data` for the given private data slot. Returns 0 if no
data was set for this object/slot combination.

This is a lightweight per-object metadata lookup with no hash table
overhead. See `create_private_data_slot` for usage patterns.
