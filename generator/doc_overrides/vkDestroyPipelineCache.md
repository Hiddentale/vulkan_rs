# Usage Notes

Destroys a pipeline cache. Pipelines that were created using this
cache remain valid, the cache is only needed during creation, not
at runtime.

Serialize the cache with `get_pipeline_cache_data` before destroying
if you want to persist it to disk for the next session.
