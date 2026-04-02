# Usage Notes

Creates an indirect execution set, a table of pipelines or
shader objects that can be indexed at execution time by
device-generated commands.

The GPU selects which pipeline/shader to use based on an index
in the generated command stream, enabling fully GPU-driven
material/shader selection.

Destroy with `destroy_indirect_execution_set_ext`.

Requires `VK_EXT_device_generated_commands`.
