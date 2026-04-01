# Usage Notes

Preprocesses device-generated commands into an intermediate
format. This can be done in a separate command buffer or pass,
then executed later with `cmd_execute_generated_commands_ext`
(with `is_preprocessed` = true).

Separating preprocessing from execution allows overlapping the
preprocessing work with other GPU tasks.

Requires `VK_EXT_device_generated_commands`.
