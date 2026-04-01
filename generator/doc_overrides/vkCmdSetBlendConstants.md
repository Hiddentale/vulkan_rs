# Usage Notes

Sets the constant blend colour used when a blend factor references
`BLEND_FACTOR_CONSTANT_COLOR`, `BLEND_FACTOR_CONSTANT_ALPHA`, or
their one-minus variants. Only takes effect if the pipeline was
created with `DYNAMIC_STATE_BLEND_CONSTANTS`.

The four values are RGBA in [0.0, 1.0]. A common use is fading
geometry by setting a constant alpha and blending with
`BLEND_FACTOR_CONSTANT_ALPHA`.

If your pipeline does not use any constant blend factors, you do not
need to set this state. The values are ignored for blend modes that
do not reference them.
