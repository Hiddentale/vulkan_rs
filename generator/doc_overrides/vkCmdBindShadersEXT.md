# Usage Notes

Binds shader objects to specified shader stages for subsequent
draw or dispatch commands. Pass arrays of stages and
corresponding shader handles.

To unbind a stage, pass a null handle for that stage. All
required stages for the draw/dispatch type must be bound.

When using shader objects, you must also set all relevant dynamic
state, there is no pipeline to provide defaults.

Requires `VK_EXT_shader_object`.
