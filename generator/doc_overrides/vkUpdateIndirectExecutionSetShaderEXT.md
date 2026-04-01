# Usage Notes

Updates entries in an indirect execution set that holds shader
objects. Each `WriteIndirectExecutionSetShaderEXT` maps an index
to a shader object handle.

The shaders must be compatible with the initial shader used to
create the execution set.

Requires `VK_EXT_device_generated_commands`.
