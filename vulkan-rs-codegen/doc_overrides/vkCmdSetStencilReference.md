# Usage Notes

Sets the stencil reference value dynamically for front-facing,
back-facing, or both face sets. Only takes effect if the pipeline
was created with `DYNAMIC_STATE_STENCIL_REFERENCE`.

The reference value is used in stencil comparison operations (e.g.
`COMPARE_OP_EQUAL` compares the masked buffer value against the
masked reference) and as the source value for `STENCIL_OP_REPLACE`.

Common patterns:

- **Portal/mirror rendering**: set reference to a unique ID per
  portal, write it with `STENCIL_OP_REPLACE`, then test with
  `COMPARE_OP_EQUAL` to mask subsequent draws to that portal's
  region.
- **Decal layering**: increment the reference per layer.
