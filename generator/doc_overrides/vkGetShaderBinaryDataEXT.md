# Usage Notes

Retrieves the binary representation of a compiled shader object.
The binary data can be cached to disk and used to create shader
objects without recompiling from SPIR-V on subsequent runs.

Call once with a null buffer to query the size, then again with
an appropriately sized buffer. Shader binaries are
implementation-specific and not portable between devices or
driver versions.

Requires `VK_EXT_shader_object`.
