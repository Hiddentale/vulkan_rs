# Usage Notes

Binds a pipeline to a command buffer for subsequent draw or dispatch
calls. The `pipeline_bind_point` must match the pipeline type:

- `PIPELINE_BIND_POINT_GRAPHICS` for graphics pipelines.
- `PIPELINE_BIND_POINT_COMPUTE` for compute pipelines.
- `PIPELINE_BIND_POINT_RAY_TRACING_KHR` for ray tracing pipelines.

Binding a pipeline invalidates any incompatible dynamic state. For
example, binding a new graphics pipeline that uses dynamic viewport
requires you to call `cmd_set_viewport` again before drawing.

Pipeline binds are relatively cheap, the driver patches command
state internally. Minimise binds by sorting draw calls by pipeline
when possible, but do not over-optimise at the expense of code
clarity.

Graphics pipelines can only be bound inside a render pass (or
dynamic rendering). Compute pipelines can be bound anywhere.

# Guide

See [Pipelines](https://hiddentale.github.io/vulkan_rs/concepts/pipelines.html) in the vulkan_rs guide.
