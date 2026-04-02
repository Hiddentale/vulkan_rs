# Usage Notes

Allocates one or more command buffers from a command pool. The
`command_buffer_count` and output slice length must match.

**Level**:

- `PRIMARY`: submitted directly to a queue via `queue_submit`.
- `SECONDARY`: recorded separately and executed inside a primary
  buffer via `cmd_execute_commands`. Useful for pre-recording
  draw calls that are reused across frames.

All allocated command buffers start in the *initial* state and must
be recorded with `begin_command_buffer` before submission.

Command buffers are freed either individually with
`free_command_buffers` or implicitly when the parent pool is
destroyed or reset.