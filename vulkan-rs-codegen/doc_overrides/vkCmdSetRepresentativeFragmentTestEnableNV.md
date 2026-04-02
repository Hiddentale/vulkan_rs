# Usage Notes

Dynamically enables or disables the representative fragment test.
When enabled, only one fragment per pixel is shaded (the
"representative" fragment), and the rest are discarded early.

Useful for visibility-only passes (e.g., occlusion culling)
where full shading is unnecessary.

Part of `VK_NV_representative_fragment_test`.

Provided by `VK_EXT_extended_dynamic_state3`.
