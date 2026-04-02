# Usage Notes

Sets the stencil compare mask dynamically for front-facing,
back-facing, or both face sets. Only takes effect if the pipeline
was created with `DYNAMIC_STATE_STENCIL_COMPARE_MASK`.

The compare mask is ANDed with both the reference value and the
stencil buffer value before the stencil comparison. This lets you
use individual bits of the stencil buffer for different purposes
(e.g. bit 0 for portals, bits 1–3 for decal layers).

A mask of `0xFF` (the default) uses all 8 bits of the stencil
buffer. Narrower masks isolate specific bit planes for multi-purpose
stencil schemes.
