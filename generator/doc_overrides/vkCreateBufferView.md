# Usage Notes

Creates a view into a buffer that interprets its contents as a
typed array of texels. Buffer views are used with:

- **Uniform texel buffers** (`BUFFER_USAGE_UNIFORM_TEXEL_BUFFER`):
  read-only typed access from shaders via `samplerBuffer` /
  `textureBuffer` in GLSL.
- **Storage texel buffers** (`BUFFER_USAGE_STORAGE_TEXEL_BUFFER`):
  read-write typed access from shaders via `imageBuffer` in GLSL.

The format, offset, and range define the view window into the
buffer. The format must be supported for the buffer view usage, 
check `format_properties.buffer_features`.

Buffer views are less common than image views. They are mainly used
for large, flat data arrays (e.g. particle attributes, lookup
tables) that benefit from format conversion on read.
