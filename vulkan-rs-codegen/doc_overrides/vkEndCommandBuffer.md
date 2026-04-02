# Usage Notes

Finishes recording a command buffer. After this call the command
buffer moves from the *recording* state to the *executable* state
and can be submitted via `queue_submit`.

If an error occurred during recording (e.g. out of memory), this
call returns the error. Always check the return value, a failed
`end_command_buffer` means the command buffer is in an invalid state
and must be reset before reuse.

A command buffer that is inside a render pass must end the render
pass with `cmd_end_render_pass` before calling `end_command_buffer`.
