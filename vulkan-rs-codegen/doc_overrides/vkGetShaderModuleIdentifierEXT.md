# Usage Notes

Retrieves the identifier for an existing shader module. The
identifier can be used for pipeline cache lookups via
`PipelineShaderStageModuleIdentifierCreateInfoEXT`.

The identifier is deterministic for the same SPIR-V content
on the same implementation.

Requires `VK_EXT_shader_module_identifier`.
