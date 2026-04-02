# Usage Notes

Resets a single command buffer back to the initial state so it can
be re-recorded. The command pool must have been created with
`RESET_COMMAND_BUFFER` for this to be valid.

**Flags**:

- `RELEASE_RESOURCES`: return the command buffer's memory to the
  pool. Without this flag, memory is retained for reuse during the
  next recording, usually preferred in a frame loop.

For bulk resets, `reset_command_pool` is more efficient than
resetting buffers individually.

The command buffer must not be pending execution when reset.
