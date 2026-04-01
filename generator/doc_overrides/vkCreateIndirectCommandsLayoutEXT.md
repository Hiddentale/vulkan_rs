# Usage Notes

Creates an indirect commands layout that defines the structure of
GPU-generated command sequences. Each token in the layout
describes one command element (draw, dispatch, push constant,
vertex buffer bind, index buffer bind, etc.).

The layout is used with `cmd_execute_generated_commands_ext`.

Destroy with `destroy_indirect_commands_layout_ext`.

Requires `VK_EXT_device_generated_commands`.
