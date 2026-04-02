# Usage Notes

A descriptor set layout defines the shape of a descriptor set: which
binding numbers exist, what descriptor type each binding holds, and
at which shader stages each binding is visible.

**Binding tips**:

- Keep `stage_flags` as narrow as possible. Declaring a binding
  visible to all stages when only the fragment shader uses it wastes
  driver resources on some implementations.
- Use `DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` for simple texture
  sampling. Separate sampler + sampled image bindings offer more
  flexibility when you want to reuse samplers across many textures.
- Array descriptors (`descriptor_count > 1`) map to GLSL arrays.
  Useful for bindless or material-table patterns.

Layouts are immutable after creation and can be shared across
multiple pipeline layouts and descriptor set allocations.

Destroy with `destroy_descriptor_set_layout` when no pipeline layout
or pending descriptor set allocation still references it.
