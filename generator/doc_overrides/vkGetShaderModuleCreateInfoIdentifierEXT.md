# Usage Notes

Computes an identifier for a shader module from its create info
(SPIR-V code) without creating the module. The identifier can be
used in `PipelineShaderStageModuleIdentifierCreateInfoEXT` to
create pipelines from cached shader data.

Useful for pipeline caching workflows where the SPIR-V is
available but you want to avoid full module creation.

Requires `VK_EXT_shader_module_identifier`.
