# Usage Notes

Dynamically enables or disables depth clamping. When enabled,
fragments outside the near/far depth range are clamped instead
of being clipped.

Useful for shadow mapping and other techniques where clipping
at the near/far plane is undesirable.

Requires the `depthClamp` device feature.

Provided by `VK_EXT_extended_dynamic_state3`.
