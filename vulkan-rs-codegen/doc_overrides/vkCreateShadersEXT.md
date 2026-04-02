# Usage Notes

Creates one or more shader objects from SPIR-V code. Shader
objects are an alternative to pipelines, instead of baking
shaders into monolithic pipeline objects, individual shader
stages can be bound independently.

Each `ShaderCreateInfoEXT` specifies the stage, code, entry
point, and optional specialization constants.

Bind with `cmd_bind_shaders_ext`. Destroy with
`destroy_shader_ext`.

Requires `VK_EXT_shader_object`.
