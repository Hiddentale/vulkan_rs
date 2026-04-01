# Usage Notes

Preprocesses device-generated commands into a form suitable for
fast execution. The preprocessing result is stored in a
preprocess buffer and later consumed by
`cmd_execute_generated_commands_nv` with `is_preprocessed` set.

Requires `VK_NV_device_generated_commands`.
