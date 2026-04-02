# Usage Notes

Executes commands generated on the GPU from an indirect commands
layout. The `GeneratedCommandsInfoEXT` specifies the indirect
commands layout, pipeline/shader objects, and the buffer
containing the generated command data.

If `is_preprocessed` is true, the command data was prepared by
a prior `cmd_preprocess_generated_commands_ext` call. Otherwise,
preprocessing and execution happen in one step.

Requires `VK_EXT_device_generated_commands`.
