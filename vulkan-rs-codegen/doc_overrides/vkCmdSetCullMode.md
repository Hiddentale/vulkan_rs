# Usage Notes

Dynamically sets the triangle culling mode. Only takes effect if
the pipeline was created with `DYNAMIC_STATE_CULL_MODE`.

Values: `CULL_MODE_NONE`, `CULL_MODE_FRONT`, `CULL_MODE_BACK`,
`CULL_MODE_FRONT_AND_BACK`.

Common pattern: set `CULL_MODE_BACK` for opaque geometry and
`CULL_MODE_NONE` for double-sided or transparent materials, without
needing separate pipelines.

Core dynamic state in Vulkan 1.3.
