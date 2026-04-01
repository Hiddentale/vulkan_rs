# Usage Notes

Creates one or more compute pipelines. Compute pipelines are simpler
than graphics pipelines, they only need a single shader stage and a
pipeline layout.

**Pipeline cache**: pass a `PipelineCache` to speed up creation, the
same way as with graphics pipelines. The cache is shared across both
pipeline types.

**Specialisation constants**: use `SpecializationInfo` on the shader
stage to bake compile-time constants into the shader (e.g. workgroup
size, algorithm variant). This produces optimised code without
duplicating shader source.

Batch multiple compute pipelines in a single call when possible.

Compute pipelines can be created at any time and are not tied to a
render pass. They are bound with `cmd_bind_pipeline` using
`PIPELINE_BIND_POINT_COMPUTE` and dispatched with `cmd_dispatch`.
