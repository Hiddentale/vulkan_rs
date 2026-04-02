# Usage Notes

Dynamically sets advanced blend parameters per color attachment.
Each `ColorBlendAdvancedEXT` specifies the advanced blend
operation, whether src/dst are premultiplied, and the blend
overlap mode (uncorrelated, disjoint, conjoint).

Only applies when using advanced blend operations (not the
standard blend factors). Requires `VK_EXT_blend_operation_advanced`.

Provided by `VK_EXT_extended_dynamic_state3`.
