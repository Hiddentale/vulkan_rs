# Usage Notes

Dynamically sets the logic operation used for color blending.
Logic ops (AND, OR, XOR, etc.) operate on the raw integer bit
patterns of the fragment and framebuffer values.

Only effective when logic op is enabled in the pipeline
(`logicOpEnable`). Requires the `extendedDynamicState2LogicOp`
feature.

Provided by `VK_EXT_extended_dynamic_state2`.
