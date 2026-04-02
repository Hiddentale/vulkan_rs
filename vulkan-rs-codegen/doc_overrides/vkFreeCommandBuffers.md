# Usage Notes

Returns individual command buffers to their parent pool. The command
buffers must not be pending execution.

For most applications, resetting the entire pool with
`reset_command_pool` is simpler. Use `free_command_buffers` only
when you need fine-grained lifetime control, for example, freeing
a one-shot transfer command buffer immediately after use while
keeping other buffers from the same pool alive.

Freed command buffer handles become invalid. Do not resubmit them.
