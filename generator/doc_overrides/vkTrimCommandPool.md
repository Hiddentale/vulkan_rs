# Usage Notes

Returns unused memory from a command pool back to the system. This
is a hint to the driver, it may or may not actually release memory.

Call this after a period of high command buffer allocation followed
by a return to lower usage (e.g. after loading screens or level
transitions). It does not affect any allocated command buffers.

Unlike `reset_command_pool`, trimming does not reset or invalidate
command buffers. It only reclaims excess internal memory that the
pool pre-allocated.

In a steady-state frame loop where you reset the pool every frame,
trimming is unnecessary, the pool reuses its memory naturally.
