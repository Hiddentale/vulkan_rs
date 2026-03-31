# Usage Notes

A command pool provides the memory backing for command buffers
allocated from it. Pools are tied to a single queue family — command
buffers allocated from the pool can only be submitted to queues of
that family.

**Flags**:

- `TRANSIENT`: hint that command buffers are short-lived and reset or
  freed frequently. Lets the driver use a faster allocation strategy.
- `RESET_COMMAND_BUFFER`: allows individual command buffers to be
  reset via `reset_command_buffer`. Without this flag you must reset
  the entire pool with `reset_command_pool`.

A common pattern is one pool per frame-in-flight per thread: reset the
whole pool at the start of each frame instead of managing individual
command buffer lifetimes.

Command pools are **not thread-safe**. If multiple threads record
commands concurrently, each thread needs its own pool.
