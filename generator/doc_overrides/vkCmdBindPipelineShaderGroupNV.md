# Usage Notes

Binds a shader group from a pipeline that was created with
multiple shader groups. Used with device-generated commands to
switch between pre-compiled shader variants without rebinding
the entire pipeline.

Requires `VK_NV_device_generated_commands`.
