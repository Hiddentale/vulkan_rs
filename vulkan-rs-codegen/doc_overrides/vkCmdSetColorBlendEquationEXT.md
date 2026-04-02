# Usage Notes

Dynamically sets the blend equation (src factor, dst factor,
blend op) for each color attachment. Each `ColorBlendEquationEXT`
specifies both the color and alpha blend parameters.

Overrides the values set at pipeline creation. Only effective for
attachments where blending is enabled.

Provided by `VK_EXT_extended_dynamic_state3`.
