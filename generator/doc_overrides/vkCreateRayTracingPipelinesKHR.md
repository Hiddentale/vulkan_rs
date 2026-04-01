# Usage Notes

Creates one or more ray tracing pipelines. A ray tracing pipeline
contains the shader stages (ray generation, miss, closest hit,
any hit, intersection, callable) and shader groups that define
how rays interact with geometry.

Unlike graphics pipelines, ray tracing pipelines organize shaders
into **groups**:

- **General**: ray generation, miss, or callable shaders.
- **Triangles hit**: closest hit + optional any hit for triangles.
- **Procedural hit**: intersection + closest hit + optional any hit
  for custom geometry (AABBs).

Pass a `DeferredOperationKHR` handle to compile asynchronously, 
the call returns `OPERATION_DEFERRED_KHR` and the pipeline handles
are not valid until the deferred operation completes. Pass a null
handle for synchronous creation.

Supports `pipeline_cache` for faster creation on subsequent runs
and `base_pipeline_handle` / `base_pipeline_index` for derivative
pipelines when `PIPELINE_CREATE_DERIVATIVE` is set.

After creation, retrieve shader group handles with
`get_ray_tracing_shader_group_handles_khr` to build the shader
binding table.
