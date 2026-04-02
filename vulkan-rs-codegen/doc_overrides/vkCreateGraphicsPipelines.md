# Usage Notes

Graphics pipeline creation is the most expensive Vulkan object creation
call. Batch multiple pipelines into a single call when possible,the
driver can often parallelise compilation internally.

**Pipeline cache**: always pass a `PipelineCache`. Even an empty cache
helps on the first run; on subsequent runs it avoids redundant shader
compilation entirely. Serialize the cache to disk between application
sessions with `get_pipeline_cache_data`.

**Dynamic state**: mark states like viewport, scissor, and blend
constants as dynamic to reduce the number of pipeline permutations.
Vulkan 1.3 makes viewport and scissor dynamic by default via
`VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT` and
`VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT`.

If creation fails for one pipeline in a batch, the call returns an
error but may still populate some output handles. Check
`VK_PIPELINE_COMPILE_REQUIRED` when using
`VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`.

# Guide

See [Pipelines](https://hiddentale.github.io/vulkan_rs/concepts/pipelines.html) in the vulkan_rs guide.
