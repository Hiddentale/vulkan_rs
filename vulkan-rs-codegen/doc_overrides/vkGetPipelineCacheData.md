# Usage Notes

Serializes the contents of a pipeline cache into a byte buffer for
storage on disk. The data includes a vendor-specific header that the
driver uses to validate compatibility on reload.

Call this with a null data pointer first to query the required buffer
size, then allocate and call again. The wrapper handles this
two-call pattern for you.

The cache data is **not portable** across different GPU vendors,
driver versions, or pipeline cache UUIDs. Always check the header
or let the driver reject incompatible data silently on reload via
`create_pipeline_cache`.

Write the data to a file (e.g. `pipeline_cache.bin`) and load it on
the next application start to avoid redundant shader compilation.
