# Usage Notes

Queries the memory requirements for preprocessing and executing
device-generated commands. Returns a `MemoryRequirements2` with
the size and alignment needed for the preprocess buffer.

Call this before allocating the preprocess buffer used by
`cmd_preprocess_generated_commands_ext` and
`cmd_execute_generated_commands_ext`.

Requires `VK_EXT_device_generated_commands`.
