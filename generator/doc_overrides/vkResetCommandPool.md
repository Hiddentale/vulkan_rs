# Usage Notes

Resets a command pool, recycling all command buffers allocated from
it back to the initial state. This is faster than resetting
individual command buffers.

**Flags**:

- `RELEASE_RESOURCES`: return memory to the system. Without this
  flag, the pool keeps its internal memory for reuse by future
  allocations, usually what you want in a frame loop.

**Per-frame pattern**: reset the pool at the start of each frame
(without `RELEASE_RESOURCES`), then re-record command buffers from
the same pool. Memory is reused without reallocation overhead.

All command buffers from this pool must have completed execution
before resetting.
