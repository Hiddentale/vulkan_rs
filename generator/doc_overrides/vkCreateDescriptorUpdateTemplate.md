# Usage Notes

Creates a template that describes how to update a descriptor set
from a compact block of host memory. Instead of building an array
of `WriteDescriptorSet` structs, you define the layout once in the
template and then call `update_descriptor_set_with_template` with
a raw pointer to your data.

**Benefits**:

- Reduces CPU overhead for frequent descriptor updates.
- Avoids allocating `WriteDescriptorSet` arrays every frame.
- Pairs well with `cmd_push_descriptor_set_with_template` for
  push descriptors.

Each entry in the template maps an offset in your host data block to
a descriptor binding, array element, and type. The driver compiles
this into an optimised update path.

Templates are immutable after creation and can be reused across
frames.
