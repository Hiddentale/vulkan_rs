# Usage Notes

Updates entries in an indirect execution set that holds pipelines.
Each `WriteIndirectExecutionSetPipelineEXT` maps an index to a
pipeline handle.

The pipelines must be compatible with the initial pipeline used
to create the execution set.

Requires `VK_EXT_device_generated_commands`.
