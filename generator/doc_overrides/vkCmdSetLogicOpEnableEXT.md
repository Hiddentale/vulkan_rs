# Usage Notes

Dynamically enables or disables logic operations for color
blending. When enabled, fragments are combined with framebuffer
values using the logic op set by `cmd_set_logic_op_ext` instead
of standard blend equations.

Logic ops operate on integer bit patterns. They have no effect
on floating-point color attachments.

Provided by `VK_EXT_extended_dynamic_state3`.
