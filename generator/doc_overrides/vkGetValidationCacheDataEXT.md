# Usage Notes

Retrieves the data from a validation cache for serialization to
disk. Call once with a null buffer to query the size, then again
with an appropriately sized buffer.

Feed the saved data back into `create_validation_cache_ext` on
the next run to avoid redundant validation.

Requires `VK_EXT_validation_cache`.
