# Usage Notes

Sets the stencil write mask dynamically for front-facing,
back-facing, or both face sets. Only takes effect if the pipeline
was created with `DYNAMIC_STATE_STENCIL_WRITE_MASK`.

The write mask controls which bits of the stencil buffer are updated
by stencil operations (`KEEP`, `REPLACE`, `INCREMENT`, etc.). Bits
that are zero in the mask are left unchanged.

A mask of `0xFF` writes all 8 bits. Use narrower masks when
different rendering passes own different stencil bit planes.
