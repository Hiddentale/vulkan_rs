# Usage Notes

Begins recording commands into a command buffer. The command buffer
must be in the *initial* state, either freshly allocated, or reset
via `reset_command_buffer` / `reset_command_pool`.

**Flags**:

- `ONE_TIME_SUBMIT`: the command buffer will be submitted once and
  then reset or freed. Lets the driver skip internal tracking it
  would otherwise need for resubmission.
- `SIMULTANEOUS_USE`: the command buffer can be pending execution on
  multiple queues simultaneously. Required for secondary command
  buffers reused across multiple primary buffers.

**Inheritance info**: only required for secondary command buffers.
When recording a secondary buffer that will execute inside a render
pass, set `render_pass`, `subpass`, and optionally `framebuffer` in
the `CommandBufferInheritanceInfo`. For primary buffers the
inheritance info is ignored.

Calling `begin_command_buffer` on a buffer that is already recording
is an error. Calling it on a buffer in the *executable* state
implicitly resets it first (if the pool allows it).

# Guide

See [Command Buffers](https://hiddentale.github.io/vulkan_rs/concepts/command-buffers.html) in the vulkan_rs guide.
