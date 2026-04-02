# Usage Notes

Creates a validation cache that stores the results of validation
layer checks. Subsequent pipeline creations with the same shaders
can skip redundant validation, improving pipeline creation time.

Provide previously saved cache data to warm-start the cache.
Retrieve data with `get_validation_cache_data_ext`.

Destroy with `destroy_validation_cache_ext`.

Requires `VK_EXT_validation_cache`.
