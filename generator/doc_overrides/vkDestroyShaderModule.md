# Usage Notes

Destroys a shader module. The module must not be in use by any
pipeline. Shader modules can be safely destroyed after pipeline
creation since the driver copies the SPIR-V at creation time.
