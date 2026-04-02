# Usage Notes

A pipeline layout defines the interface between shader stages and
the descriptor sets and push constants that feed them. It specifies:

- **Descriptor set layouts**: which bindings are available at each
  set index (0, 1, 2, ...).
- **Push constant ranges**: byte ranges per shader stage for small,
  frequently-updated data.

**Set layout ordering convention**: a common pattern is:

- Set 0: per-frame data (camera, time).
- Set 1: per-material data (textures, material params).
- Set 2: per-object data (transforms).

This lets you bind set 0 once per frame and only rebind sets 1–2
as materials and objects change, minimising descriptor set switches.

Pipeline layouts are immutable after creation. Two pipelines that
share the same layout can share descriptor sets without rebinding.

Push constants are limited to `max_push_constants_size` bytes
(guaranteed at least 128). Use them for small per-draw data like
transform matrices or material indices.
