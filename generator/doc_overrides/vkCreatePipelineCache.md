# Usage Notes

Creates a pipeline cache that stores compiled pipeline state to
speed up future pipeline creation.

**Initial data**: pass previously serialized cache data (from
`get_pipeline_cache_data`) to warm the cache on startup. The driver
validates the header and silently ignores data from incompatible
driver versions or hardware, it is always safe to pass stale data.

A single cache can be shared across all pipeline creation calls in
the application. Multiple threads can use the same cache
concurrently, the driver handles internal synchronization.

**Recommended workflow**:

1. On startup, load cache data from disk and create the cache.
2. Pass the cache to every `create_graphics_pipelines` and
   `create_compute_pipelines` call.
3. On shutdown, serialize with `get_pipeline_cache_data` and write
   to disk.
